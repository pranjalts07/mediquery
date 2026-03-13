"""
app/rag.py
Retrieval-Augmented Generation pipeline with conversation memory.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx
from pinecone import Pinecone

from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MediQuery, a warm, knowledgeable medical information assistant with the bedside manner of a trusted doctor friend.

Your style:
- Speak naturally and conversationally — never say "based on the provided context", "according to the context", "from the context", or any variation of this. Just answer confidently and directly.
- Use plain English. Avoid overly clinical language unless necessary, and explain medical terms when you use them.
- Be warm, reassuring, and empathetic — especially when someone is worried about symptoms.
- Structure longer answers with short paragraphs. Use bullet points only when listing multiple distinct items.
- Always end with a gentle recommendation to see a doctor for personal or serious concerns.
- If you don't have enough information to answer well, say so honestly and suggest they consult a healthcare professional.
- Never diagnose, never prescribe, never replace professional medical advice.
- If the user refers to something they mentioned earlier in the conversation, use that context to give a better answer."""

USER_TEMPLATE = """Medical knowledge:
{context}

Patient question: {question}

Answer naturally and helpfully as a knowledgeable medical assistant. Do not reference or mention the medical knowledge above — just use it to inform your answer."""


def embed_query(text: str, settings: Settings) -> list[float]:
    url = f"https://router.huggingface.co/hf-inference/models/{settings.hf_embedding_model}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        logger.error("HF embedding error %s: %s", resp.status_code, resp.text[:400])
        raise RuntimeError(f"HF embedding API returned {resp.status_code}. Check HF_API_TOKEN and model availability.")

    result = resp.json()
    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]
    if isinstance(result, list) and isinstance(result[0], float):
        return result
    raise ValueError(f"Unexpected embedding response shape: {type(result)}")


@lru_cache(maxsize=1)
def _get_pinecone_index(api_key: str, host: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(host=host)


def retrieve(query_vector: list[float], settings: Settings) -> list[dict[str, Any]]:
    index = _get_pinecone_index(settings.pinecone_api_key, settings.pinecone_host)
    response = index.query(vector=query_vector, top_k=settings.top_k, include_metadata=True)
    results = []
    for match in response.get("matches", []):
        metadata = match.get("metadata", {})
        results.append({
            "text": metadata.get("text", ""),
            "source": metadata.get("source", "MediQuery Knowledge Base"),
            "score": round(float(match.get("score", 0.0)), 4),
        })
    return results


def generate(prompt: str, settings: Settings, history: list[dict] | None = None) -> str:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.hf_api_token}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for msg in history[-6:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.hf_llm_model,
        "messages": messages,
        "max_tokens": settings.max_new_tokens,
        "temperature": 0.4,
        "stream": False,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        logger.error("HF generation error %s: %s", resp.status_code, resp.text[:400])
        raise RuntimeError(f"HF generation API returned {resp.status_code}. Check HF_API_TOKEN and model availability.")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected generation response structure: {data}") from e


def run_rag(question: str, settings: Settings, history: list[dict] | None = None) -> dict[str, Any]:
    logger.info("RAG query: %s", question[:120])

    query_vector = embed_query(question, settings)
    chunks = retrieve(query_vector, settings)
    logger.info("Retrieved %d chunks from Pinecone", len(chunks))

    if not chunks:
        return {
            "answer": "I don't have specific information on that in my knowledge base. For accurate guidance on this topic, I'd recommend speaking with a qualified healthcare professional.",
            "sources": [],
        }

    context_text = "\n\n---\n\n".join(f"{chunk['text']}" for chunk in chunks)
    user_prompt = USER_TEMPLATE.format(context=context_text, question=question)

    answer = generate(user_prompt, settings, history=history)

    sources = [
        {"source": chunk["source"], "score": chunk["score"], "text": chunk["text"]}
        for chunk in chunks if chunk["text"]
    ]

    return {"answer": answer, "sources": sources}