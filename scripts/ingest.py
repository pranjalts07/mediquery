"""
scripts/ingest.py
Ingestion pipeline for MediQuery knowledge base.

Usage:
    python scripts/ingest.py [--data data/sample_knowledge.jsonl] [--batch-size 32]

What it does:
  1. Reads documents from a .jsonl file (one JSON object per line)
  2. Embeds each document's text using the HF Inference API
     (sentence-transformers/all-MiniLM-L6-v2 → 384-dim vectors)
  3. Upserts all vectors into Pinecone in batches

Each JSONL line must have at minimum:
    { "id": "unique-id", "text": "document text", "source": "source name" }

Prerequisites:
    pip install -r requirements.txt
    Set environment variables (see .env.example)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest")

# ── Config (from env) ─────────────────────────────────────────────────────────

HF_API_TOKEN = os.environ["HF_API_TOKEN"]
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_HOST = os.environ["PINECONE_HOST"]

EMBED_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], max_retries: int = 4) -> list[list[float]]:
    """
    Embed a batch of texts in a single HF Inference API call.
    Retries with exponential backoff on timeout or 503.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(EMBED_URL, json=payload, headers=headers)

            if resp.status_code in (503, 504):
                wait = 2 ** attempt
                logger.warning("HF API %d (attempt %d/%d) — retrying in %ds", resp.status_code, attempt, max_retries, wait)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                logger.error("Embedding API error %s: %s", resp.status_code, resp.text[:300])
                raise RuntimeError(f"HF Inference API returned {resp.status_code}")

            result = resp.json()
            if isinstance(result, list) and result and isinstance(result[0], list):
                return result
            if isinstance(result, list) and result and isinstance(result[0], float):
                return [result]
            raise ValueError(f"Unexpected embedding response shape: {type(result)}")

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            wait = 2 ** attempt
            logger.warning("Timeout on attempt %d/%d — retrying in %ds (%s)", attempt, max_retries, wait, exc)
            time.sleep(wait)

    raise RuntimeError(f"Embedding failed after {max_retries} attempts")


# ── Pinecone upsert ───────────────────────────────────────────────────────────

def upsert_batch(index, records: list[dict]) -> None:
    vectors = []
    for rec in records:
        vectors.append(
            {
                "id": rec["id"],
                "values": rec["embedding"],
                "metadata": {
                    "text": rec["text"],
                    "source": rec.get("source", "MediQuery Knowledge Base"),
                },
            }
        )
    index.upsert(vectors=vectors)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str, batch_size: int) -> None:
    path = Path(data_path)
    if not path.exists():
        logger.error("Data file not found: %s", path)
        sys.exit(1)

    # Load documents
    docs = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                assert "id" in doc and "text" in doc, "Each record needs 'id' and 'text'"
                docs.append(doc)
            except (json.JSONDecodeError, AssertionError) as e:
                logger.warning("Skipping line %d: %s", line_no, e)

    logger.info("Loaded %d documents from %s", len(docs), path)

    if not docs:
        logger.error("No valid documents found. Exiting.")
        sys.exit(1)

    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    logger.info("Connected to Pinecone index: %s", PINECONE_INDEX_NAME)

    # Process in batches
    total_upserted = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        # all-MiniLM-L6-v2 max is 256 tokens (~200 words). Truncate to avoid timeouts.
        texts = [" ".join(d["text"].split()[:200]) for d in batch]

        logger.info(
            "Embedding batch %d–%d of %d...",
            start + 1,
            min(start + batch_size, len(docs)),
            len(docs),
        )

        embeddings = embed_texts(texts)

        records = [
            {**doc, "embedding": emb}
            for doc, emb in zip(batch, embeddings)
        ]

        upsert_batch(index, records)
        total_upserted += len(records)

        logger.info("Upserted %d/%d vectors", total_upserted, len(docs))

        # Polite rate-limit pause between batches
        if start + batch_size < len(docs):
            time.sleep(0.5)

    logger.info("✅ Ingestion complete. %d vectors stored in Pinecone.", total_upserted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediQuery knowledge base ingestion")
    parser.add_argument(
        "--data",
        default="data/combined_knowledge.jsonl",
        help="Path to .jsonl data file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of texts to embed per API call (default: 16)",
    )
    args = parser.parse_args()
    main(args.data, args.batch_size)
