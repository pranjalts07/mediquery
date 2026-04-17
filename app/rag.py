from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from functools import lru_cache
from typing import Any, AsyncIterator

import httpx
from pinecone import Pinecone

from app.circuit_breaker import CircuitBreaker
from app.config import Settings
from app.metrics import metrics

logger = logging.getLogger(__name__)

MIN_SCORE = 0.35
RERANKER_TOP_N = 3
MAX_CONTEXT_CHARS = 12_000

_embed_breaker = CircuitBreaker("embedding", failure_threshold=5, recovery_timeout=60.0)
_generate_breaker = CircuitBreaker("generation", failure_threshold=5, recovery_timeout=60.0)
_rerank_breaker = CircuitBreaker("reranker", failure_threshold=5, recovery_timeout=60.0)

_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.ConnectError)

# In-memory embedding cache: key → (vector, monotonic timestamp)
_EMBED_CACHE: dict[str, tuple[list[float], float]] = {}
_CACHE_MAX = 256
_CACHE_TTL = 300.0

SYSTEM_PROMPT = """You are MediQuery, a warm, knowledgeable medical information assistant with the bedside manner of a trusted doctor friend.

Your style:
- Speak naturally and conversationally — never say "based on the provided context", "according to the context", "from the context", or any variation of this. Just answer confidently and directly.
- Use plain English. Avoid overly clinical language unless necessary, and explain medical terms when you use them.
- Be warm, reassuring, and empathetic — especially when someone is worried about symptoms.
- Structure longer answers with short paragraphs. Use bullet points only when listing multiple distinct items.
- Only recommend seeing a doctor when the question is about personal symptoms, diagnosis, or treatment decisions — not for general knowledge questions.
- If the medical knowledge provided does not cover the question, say so honestly and suggest they consult a healthcare professional. Do not fill gaps with information that isn't in the provided knowledge.
- Never diagnose, never prescribe, never replace professional medical advice.
- If the user refers to something they mentioned earlier in the conversation, use that context to give a better answer.

Critical rule: Base your answer strictly on the medical knowledge provided. Do not add facts, statistics, or claims that are not present in the provided knowledge."""

_MODE_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Respond in 2-3 sentences maximum. Give the essential point only. "
        "Do not elaborate, do not add caveats, do not add follow-up suggestions. Stop after the key fact."
    ),
    "detailed": (
        "Answer naturally and helpfully as a knowledgeable medical assistant. "
        "Do not reference or mention the medical knowledge above — just use it to inform your answer."
    ),
}


def _build_user_prompt(context: str, question: str, instruction: str) -> str:
    return f"Medical knowledge:\n{context}\n\nPatient question: {question}\n\n{instruction}"


def _embed_cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:16]


def _cache_lookup(text: str, model: str) -> list[float] | None:
    key = _embed_cache_key(text, model)
    entry = _EMBED_CACHE.get(key)
    if entry and time.monotonic() - entry[1] < _CACHE_TTL:
        return entry[0]
    _EMBED_CACHE.pop(key, None)
    return None


def _cache_store(text: str, model: str, embedding: list[float]) -> None:
    key = _embed_cache_key(text, model)
    if len(_EMBED_CACHE) >= _CACHE_MAX:
        del _EMBED_CACHE[next(iter(_EMBED_CACHE))]
    _EMBED_CACHE[key] = (embedding, time.monotonic())


async def _do_embed(text: str, settings: Settings) -> list[float]:
    url = f"https://router.huggingface.co/hf-inference/models/{settings.hf_embedding_model}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            break
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < 2:
                logger.warning("Embedding timeout (attempt %d/3), retrying...", attempt + 1)
                await asyncio.sleep(2**attempt)
    else:
        raise RuntimeError("Embedding service unavailable after 3 attempts.") from last_exc

    if resp.status_code != 200:
        logger.error("HF embedding API error: status=%s", resp.status_code)
        raise RuntimeError(f"HF embedding API returned {resp.status_code}.")

    result = resp.json()
    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]
    if isinstance(result, list) and isinstance(result[0], float):
        return result
    raise ValueError(f"Unexpected embedding response shape: {type(result)}")


async def embed_query(text: str, settings: Settings) -> list[float]:
    cached = _cache_lookup(text, settings.hf_embedding_model)
    if cached is not None:
        metrics.record_cache_hit()
        logger.debug("Embedding cache hit")
        return cached

    metrics.record_cache_miss()
    embedding = await _embed_breaker.call(_do_embed(text, settings))
    _cache_store(text, settings.hf_embedding_model, embedding)
    return embedding


@lru_cache(maxsize=1)
def _get_pinecone_index(api_key: str, host: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(host=host)


async def retrieve(query_vector: list[float], settings: Settings) -> list[dict[str, Any]]:
    index = _get_pinecone_index(settings.pinecone_api_key, settings.pinecone_host)
    response = await asyncio.to_thread(
        index.query,
        vector=query_vector,
        top_k=settings.top_k,
        include_metadata=True,
    )
    results = []
    for match in response.get("matches", []):
        score = round(float(match.get("score", 0.0)), 4)
        if score < MIN_SCORE:
            logger.debug("Dropping low-score chunk (score=%.4f < threshold=%.2f)", score, MIN_SCORE)
            continue
        metadata = match.get("metadata", {})
        results.append({
            "text":   metadata.get("text", ""),
            "source": metadata.get("source", "MediQuery Knowledge Base"),
            "score":  score,
        })
    return results


async def rerank(question: str, chunks: list[dict], settings: Settings) -> list[dict]:
    if len(chunks) <= RERANKER_TOP_N:
        return chunks

    url = f"https://router.huggingface.co/hf-inference/models/{settings.hf_reranker_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": [[question, chunk["text"]] for chunk in chunks]}

    async def _do_rerank() -> list[dict]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code != 200:
            logger.warning("Reranker API error %s — using original Pinecone order", resp.status_code)
            return chunks[:RERANKER_TOP_N]

        raw = resp.json()
        if raw and isinstance(raw[0], (int, float)):
            scores = [float(s) for s in raw]
        elif raw and isinstance(raw[0], list):
            scores = [max(d.get("score", 0.0) for d in item) for item in raw]
        else:
            logger.warning("Unexpected reranker response shape — using original Pinecone order")
            return chunks[:RERANKER_TOP_N]

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        reranked = [chunk for chunk, _ in ranked[:RERANKER_TOP_N]]
        logger.info("Reranker: %d → %d chunks", len(chunks), len(reranked))
        return reranked

    try:
        return await _rerank_breaker.call(_do_rerank())
    except Exception as exc:
        logger.warning("Reranker failed (%s) — using original Pinecone order", exc)
        return chunks[:RERANKER_TOP_N]


def _build_messages(prompt: str, history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history[-6:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _do_generate(prompt: str, settings: Settings, history: list[dict] | None) -> str:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       settings.hf_llm_model,
        "messages":    _build_messages(prompt, history),
        "max_tokens":  settings.max_new_tokens,
        "temperature": 0.4,
        "stream":      False,
    }

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            break
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < 2:
                logger.warning("Generation timeout (attempt %d/3), retrying...", attempt + 1)
                await asyncio.sleep(2**attempt)
    else:
        raise RuntimeError("Generation service unavailable after 3 attempts.") from last_exc

    if resp.status_code != 200:
        logger.error("HF generation API error: status=%s", resp.status_code)
        raise RuntimeError(f"HF generation API returned {resp.status_code}.")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected generation response structure: {data}") from exc


async def generate(prompt: str, settings: Settings, history: list[dict] | None = None) -> str:
    return await _generate_breaker.call(_do_generate(prompt, settings, history))


async def generate_stream(
    prompt: str, settings: Settings, history: list[dict] | None = None
) -> AsyncIterator[str]:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       settings.hf_llm_model,
        "messages":    _build_messages(prompt, history),
        "max_tokens":  settings.max_new_tokens,
        "temperature": 0.4,
        "stream":      True,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                logger.error("HF streaming API error: status=%s", resp.status_code)
                raise RuntimeError(f"HF generation API returned {resp.status_code}.")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def _retrieval_query(question: str, history: list[dict] | None) -> str:
    words = question.strip().split()
    if len(words) <= 5 and history:
        last_user = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), None
        )
        if last_user:
            return f"{last_user} — {question}"
    return question


def get_circuit_breaker_states() -> dict[str, str]:
    return {
        "embedding":  _embed_breaker.state_name,
        "generation": _generate_breaker.state_name,
        "reranker":   _rerank_breaker.state_name,
    }


def _build_context(chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    if len(context) > MAX_CONTEXT_CHARS:
        logger.warning("Context truncated from %d to %d chars", len(context), MAX_CONTEXT_CHARS)
        context = context[:MAX_CONTEXT_CHARS]
    return context


async def run_rag(
    question: str,
    settings: Settings,
    history: list[dict] | None = None,
    mode: str = "detailed",
) -> dict[str, Any]:
    logger.info("RAG query received (length=%d)", len(question))
    t0 = time.perf_counter()

    retrieval_q = _retrieval_query(question, history)
    query_vector = await embed_query(retrieval_q, settings)
    chunks = await retrieve(query_vector, settings)
    logger.info("Retrieved %d chunks from Pinecone (above score threshold)", len(chunks))

    if not chunks:
        return {
            "answer": (
                "I don't have specific information on that in my knowledge base. "
                "For accurate guidance on this topic, I'd recommend speaking with "
                "a qualified healthcare professional."
            ),
            "sources": [],
        }

    chunks = await rerank(question, chunks, settings)
    context = _build_context(chunks)
    instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["detailed"])
    user_prompt = _build_user_prompt(context, question, instruction)
    answer = await generate(user_prompt, settings, history=history)

    latency_ms = (time.perf_counter() - t0) * 1000
    metrics.record_request(latency_ms)

    sources = [
        {"source": chunk["source"], "score": chunk["score"], "text": chunk["text"]}
        for chunk in chunks
        if chunk["text"]
    ]
    return {"answer": answer, "sources": sources}


async def run_rag_stream(
    question: str,
    settings: Settings,
    history: list[dict] | None = None,
    mode: str = "detailed",
) -> AsyncIterator[str]:
    logger.info("RAG stream query received (length=%d)", len(question))
    t0 = time.perf_counter()

    try:
        retrieval_q = _retrieval_query(question, history)
        query_vector = await embed_query(retrieval_q, settings)
        chunks = await retrieve(query_vector, settings)
        logger.info("Retrieved %d chunks (stream path)", len(chunks))

        if not chunks:
            no_info = (
                "I don't have specific information on that in my knowledge base. "
                "For accurate guidance on this topic, I'd recommend speaking with "
                "a qualified healthcare professional."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': no_info})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            metrics.record_request((time.perf_counter() - t0) * 1000)
            return

        chunks = await rerank(question, chunks, settings)
        sources = [
            {"source": c["source"], "score": c["score"], "text": c["text"]}
            for c in chunks
            if c["text"]
        ]

        context = _build_context(chunks)
        instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["detailed"])
        user_prompt = _build_user_prompt(context, question, instruction)

        async for token in generate_stream(user_prompt, settings, history=history):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"
        metrics.record_request((time.perf_counter() - t0) * 1000)

    except RuntimeError as exc:
        logger.error("RAG stream pipeline error: %s", exc)
        metrics.record_request((time.perf_counter() - t0) * 1000, error=True)
        yield f"data: {json.dumps({'type': 'error', 'content': 'The AI service is temporarily unavailable. Please try again shortly.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
    except Exception:
        logger.exception("Unexpected error in RAG stream pipeline")
        metrics.record_request((time.perf_counter() - t0) * 1000, error=True)
        yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
