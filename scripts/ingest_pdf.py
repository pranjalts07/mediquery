"""scripts/ingest_pdf.py — Convert a medical PDF into a JSONL knowledge base.

Usage:
    python scripts/ingest_pdf.py --pdf path/to/book.pdf --out data/knowledge_base.jsonl

Then run:
    python scripts/ingest.py --data data/knowledge_base.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest_pdf")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
MIN_CHUNK_WORDS = 40
SKIP_PAGES = 14

NOISE_PATTERNS = [
    r"^GALE ENCYCLOPEDIA",
    r"^Gale Encyclopedia",
    r"^\d+\s*$",
    r"^[A-Z\s]{2,50}$",
    r"^\s*[ivxlcdmIVXLCDM]+\s*$",
    r"^KEY TERMS$",
    r"^ORGANIZATIONS$",
    r"^BOOKS$",
    r"^PERIODICALS$",
    r"^Resources$",
    r"^Further Reading",
    r".*\(\d{3}\)\s*\d{3}-\d{4}.*",
    r".*<http://.*>.*",
    r".*FAX:.*",
    r"Medical Writer$",
    r"Medical Writer\n",
]
NOISE_RE = [re.compile(p) for p in NOISE_PATTERNS]


def clean_text(raw: str) -> str:
    # Fix hyphenated line breaks
    text = re.sub(r"-\n([a-z])", r"\1", raw)
    # Join mid-sentence line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalise multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace bullet characters
    text = re.sub(r"[•·▪◦‣⁃]", "-", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if any(p.match(stripped) for p in NOISE_RE):
            continue
        cleaned.append(stripped)

    text = "\n".join(cleaned)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-count chunks, breaking on sentence boundaries where possible."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if end < len(words):
            search_text = chunk
            last_sent = max(
                search_text.rfind(". "),
                search_text.rfind("? "),
                search_text.rfind("! "),
            )
            if last_sent > len(chunk) // 2:
                chunk = chunk[: last_sent + 1].strip()

        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def main(pdf_path: str, out_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> None:
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", path)
        sys.exit(1)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading PDF: %s", path)
    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    logger.info("Total pages: %d (skipping first %d)", total_pages, SKIP_PAGES)

    full_text_parts = []
    for i, page in enumerate(reader.pages):
        if i < SKIP_PAGES:
            continue
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            full_text_parts.append(cleaned)

        if (i + 1) % 100 == 0:
            logger.info("Extracted %d/%d pages...", i + 1, total_pages)

    full_text = "\n\n".join(full_text_parts)
    logger.info("Extraction complete. Total words: %d", len(full_text.split()))

    chunks = chunk_text(full_text, chunk_size, overlap)
    logger.info("Generated %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)

    written = 0
    skipped = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            if word_count < MIN_CHUNK_WORDS:
                skipped += 1
                continue

            record = {
                "id": f"gale-med-{i:05d}",
                "text": chunk,
                "source": "Gale Encyclopedia of Medicine",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Done. Written: %d chunks, Skipped: %d (too short)", written, skipped)
    logger.info("Output: %s", out)
    logger.info("")
    logger.info("Next step — push to Pinecone:")
    logger.info("  python scripts/ingest.py --data %s --batch-size 16", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a medical PDF into a JSONL knowledge base for MediQuery"
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--out",
        default="data/knowledge_base.jsonl",
        help="Output JSONL path (default: data/knowledge_base.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Words per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Word overlap between chunks (default: {CHUNK_OVERLAP})",
    )
    args = parser.parse_args()
    main(args.pdf, args.out, chunk_size=args.chunk_size, overlap=args.overlap)