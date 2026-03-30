"""
app/rag.py
Retrieval-Augmented Generation pipeline with conversation memory.

All I/O is fully async:
  - HF API calls use httpx.AsyncClient (never blocks the event loop)
  - Pinecone index.query() is synchronous, wrapped in asyncio.to_thread()

Two entry points:
  run_rag()        → returns a dict {answer, sources} — used by POST /chat
  run_rag_stream() → async generator that yields SSE-formatted strings — used by POST /chat/stream

Security notes:
  - API keys are passed via the Settings object; never logged or returned to callers.
  - Error responses from upstream APIs are logged at status-code level only.
  - The Pinecone client is cached (lru_cache). Cache key includes api_key so a
    key rotation clears the cache.
"""
from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any, AsyncIterator

import httpx
from pinecone import Pinecone

from app.config import Settings

logger = logging.getLogger(__name__)

# Minimum cosine similarity a Pinecone result must have to be used.
MIN_SCORE = 0.35

# Cross-encoder keeps this many chunks after reranking.
RERANKER_TOP_N = 3

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

USER_TEMPLATE = """Medical knowledge:
{context}

Patient question: {question}

{instruction}"""


# ── Embedding ──────────────────────────────────────────────────────────────────

async def embed_query(text: str, settings: Settings) -> list[float]:
    url = f"https://router.huggingface.co/hf-inference/models/{settings.hf_embedding_model}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        logger.error("HF embedding API error: status=%s", resp.status_code)
        raise RuntimeError(f"HF embedding API returned {resp.status_code}.")

    result = resp.json()
    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]
    if isinstance(result, list) and isinstance(result[0], float):
        return result
    raise ValueError(f"Unexpected embedding response shape: {type(result)}")


# ── Retrieval ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_pinecone_index(api_key: str, host: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(host=host)


async def retrieve(query_vector: list[float], settings: Settings) -> list[dict[str, Any]]:
    index = _get_pinecone_index(settings.pinecone_api_key, settings.pinecone_host)
    # Pinecone client is synchronous — run in a thread so we don't block the event loop.
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


# ── Reranking ──────────────────────────────────────────────────────────────────

async def rerank(question: str, chunks: list[dict], settings: Settings) -> list[dict]:
    """
    Cross-encoder reranking via HF Inference API.

    Bi-encoder (Pinecone) optimises for speed and recall — it finds roughly
    the right documents but can mis-order them. A cross-encoder sees the full
    (query, passage) pair and scores relevance far more accurately.

    We retrieve top_k=8 broad candidates then rerank to RERANKER_TOP_N=3.
    Falls back to the original Pinecone order if the API call fails.
    """
    if len(chunks) <= RERANKER_TOP_N:
        return chunks

    url = f"https://router.huggingface.co/hf-inference/models/{settings.hf_reranker_model}"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": [[question, chunk["text"]] for chunk in chunks]}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code != 200:
            logger.warning(
                "Reranker API error %s — using original Pinecone order", resp.status_code
            )
            return chunks[:RERANKER_TOP_N]

        raw = resp.json()

        # Parse two common response shapes:
        # Shape A — flat floats:        [0.92, 0.41, ...]
        # Shape B — label/score dicts:  [[{"label": "LABEL_1", "score": 0.92}], ...]
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

    except Exception as exc:
        logger.warning("Reranker failed (%s) — using original Pinecone order", exc)
        return chunks[:RERANKER_TOP_N]


# ── Generation ─────────────────────────────────────────────────────────────────

def _build_messages(prompt: str, history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history[-6:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    return messages


async def generate(prompt: str, settings: Settings, history: list[dict] | None = None) -> str:
    """Single-shot generation — returns the full answer as a string."""
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        logger.error("HF generation API error: status=%s", resp.status_code)
        raise RuntimeError(f"HF generation API returned {resp.status_code}.")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected generation response structure: {data}") from exc


async def generate_stream(
    prompt: str, settings: Settings, history: list[dict] | None = None
) -> AsyncIterator[str]:
    """
    Streaming generation — yields raw text tokens as they arrive from the HF API.

    The HF chat completions endpoint returns standard OpenAI-compatible SSE:
      data: {"choices": [{"delta": {"content": "token"}}]}
      data: [DONE]
    """
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


# ── RAG entry points ───────────────────────────────────────────────────────────

async def run_rag(
    question: str,
    settings: Settings,
    history: list[dict] | None = None,
    mode: str = "detailed",
) -> dict[str, Any]:
    """Full RAG pipeline — returns {answer, sources}. Used by POST /chat."""
    logger.info("RAG query received (length=%d)", len(question))

    query_vector = await embed_query(question, settings)
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

    context_text = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["detailed"])
    user_prompt = USER_TEMPLATE.format(
        context=context_text, question=question, instruction=instruction
    )

    answer = await generate(user_prompt, settings, history=history)

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
    """
    Streaming RAG pipeline — yields SSE-formatted strings.

    Event types:
      {"type": "token",  "content": "..."}     — one token of the answer
      {"type": "done",   "sources": [...]}      — generation complete + sources
      {"type": "error",  "content": "..."}      — pipeline error
    """
    logger.info("RAG stream query received (length=%d)", len(question))

    try:
        query_vector = await embed_query(question, settings)
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
            return

        chunks = await rerank(question, chunks, settings)

        sources = [
            {"source": c["source"], "score": c["score"], "text": c["text"]}
            for c in chunks
            if c["text"]
        ]

        context_text = "\n\n---\n\n".join(c["text"] for c in chunks)
        instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["detailed"])
        user_prompt = USER_TEMPLATE.format(
            context=context_text, question=question, instruction=instruction
        )

        async for token in generate_stream(user_prompt, settings, history=history):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

    except RuntimeError as exc:
        logger.error("RAG stream pipeline error: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'content': 'The AI service is temporarily unavailable. Please try again shortly.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
    except Exception:
        logger.exception("Unexpected error in RAG stream pipeline")
        yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
