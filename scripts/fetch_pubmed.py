"""scripts/fetch_pubmed.py — Fetch PubMed abstracts as a JSONL knowledge base.

No API key required — PubMed's E-utilities API is free and open.

Usage:
    python scripts/fetch_pubmed.py
    python scripts/fetch_pubmed.py --out data/pubmed_knowledge.jsonl

Then ingest into Pinecone:
    python scripts/ingest.py --data data/pubmed_knowledge.jsonl --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fetch_pubmed")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Each entry: (search_query, max_results, friendly_label)
TOPICS = [
    # Cardiovascular
    ("hypertension treatment guidelines[Title/Abstract] hasabstract",           40, "Hypertension"),
    ("heart failure management clinical[Title/Abstract] hasabstract",           40, "Heart Failure"),
    ("atrial fibrillation treatment[Title/Abstract] hasabstract",               30, "Atrial Fibrillation"),
    ("coronary artery disease prevention[Title/Abstract] hasabstract",          30, "Coronary Artery Disease"),

    # Metabolic
    ("type 2 diabetes mellitus treatment[Title/Abstract] hasabstract",          40, "Type 2 Diabetes"),
    ("insulin resistance pathophysiology[Title/Abstract] hasabstract",          25, "Insulin Resistance"),
    ("obesity management guidelines[Title/Abstract] hasabstract",               30, "Obesity"),
    ("hyperlipidemia statin therapy[Title/Abstract] hasabstract",               30, "Hyperlipidemia"),

    # Respiratory
    ("asthma management inhaled corticosteroids[Title/Abstract] hasabstract",   30, "Asthma"),
    ("COPD chronic obstructive pulmonary disease treatment[Title/Abstract] hasabstract", 30, "COPD"),
    ("pneumonia community acquired treatment[Title/Abstract] hasabstract",      25, "Pneumonia"),

    # Infectious Disease
    ("antibiotic resistance antimicrobial stewardship[Title/Abstract] hasabstract", 30, "Antimicrobial Resistance"),
    ("sepsis management clinical[Title/Abstract] hasabstract",                  25, "Sepsis"),
    ("HIV antiretroviral therapy[Title/Abstract] hasabstract",                  25, "HIV"),
    ("COVID-19 treatment outcomes[Title/Abstract] hasabstract",                 30, "COVID-19"),

    # Neurology / Mental Health
    ("depression antidepressant treatment[Title/Abstract] hasabstract",         30, "Depression"),
    ("anxiety disorder cognitive behavioral therapy[Title/Abstract] hasabstract", 25, "Anxiety"),
    ("stroke prevention treatment[Title/Abstract] hasabstract",                 30, "Stroke"),
    ("Alzheimer disease treatment[Title/Abstract] hasabstract",                 25, "Alzheimer's Disease"),
    ("migraine treatment prevention[Title/Abstract] hasabstract",               25, "Migraine"),

    # Oncology
    ("breast cancer treatment outcomes[Title/Abstract] hasabstract",            30, "Breast Cancer"),
    ("lung cancer immunotherapy[Title/Abstract] hasabstract",                   25, "Lung Cancer"),
    ("colorectal cancer screening prevention[Title/Abstract] hasabstract",      25, "Colorectal Cancer"),

    # Gastroenterology
    ("irritable bowel syndrome treatment[Title/Abstract] hasabstract",          25, "IBS"),
    ("gastroesophageal reflux disease GERD treatment[Title/Abstract] hasabstract", 25, "GERD"),

    # Musculoskeletal
    ("rheumatoid arthritis treatment biologics[Title/Abstract] hasabstract",    25, "Rheumatoid Arthritis"),
    ("osteoporosis prevention treatment[Title/Abstract] hasabstract",           25, "Osteoporosis"),
    ("low back pain management[Title/Abstract] hasabstract",                    25, "Low Back Pain"),

    # Endocrine
    ("thyroid disease hypothyroidism treatment[Title/Abstract] hasabstract",    25, "Thyroid Disease"),
    ("type 1 diabetes insulin therapy[Title/Abstract] hasabstract",             25, "Type 1 Diabetes"),

    # Preventive / General
    ("vaccination immunization efficacy[Title/Abstract] hasabstract",           25, "Vaccination"),
    ("cancer screening recommendations[Title/Abstract] hasabstract",            25, "Cancer Screening"),
    ("sleep disorders insomnia treatment[Title/Abstract] hasabstract",          25, "Sleep Disorders"),
    ("pain management analgesics clinical[Title/Abstract] hasabstract",         25, "Pain Management"),
    ("nutrition diet cardiovascular health[Title/Abstract] hasabstract",        25, "Nutrition"),
]


def search_pmids(query: str, max_results: int, client: httpx.Client) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = client.get(ESEARCH_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str], client: httpx.Client) -> list[dict]:
    """Fetch full abstract records for a list of PMIDs. Returns list of dicts."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = client.get(EFETCH_URL, params=params)
    resp.raise_for_status()

    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)

    records = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID", "").strip()
            title = article.findtext(".//ArticleTitle", "").strip()

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.get("Label", "") + ": " if p.get("Label") else "") + (p.text or "")
                for p in abstract_parts
            ).strip()

            if not abstract or len(abstract.split()) < 30:
                continue

            authors = []
            for author in article.findall(".//Author")[:3]:
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())
            author_str = ", ".join(authors)
            if len(article.findall(".//Author")) > 3:
                author_str += " et al."

            journal = article.findtext(".//Journal/Title", "") or \
                      article.findtext(".//Journal/ISOAbbreviation", "")
            year = article.findtext(".//PubDate/Year", "") or \
                   article.findtext(".//PubDate/MedlineDate", "")[:4]

            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": author_str,
                "journal": journal,
                "year": year,
            })

        except Exception as e:
            logger.debug("Skipping article parse error: %s", e)
            continue

    return records


def build_record(rec: dict, topic_label: str) -> dict:
    """Convert a PubMed record into a MediQuery JSONL record."""
    text = f"{rec['title']}. {rec['abstract']}"

    citation_parts = []
    if rec["authors"]:
        citation_parts.append(rec["authors"])
    if rec["journal"]:
        citation_parts.append(rec["journal"])
    if rec["year"]:
        citation_parts.append(rec["year"])
    citation = ". ".join(citation_parts)

    source = f"{rec['title']} — PMID {rec['pmid']}"
    if citation:
        source += f" | {citation}"

    return {
        "id": f"pubmed-{rec['pmid']}",
        "text": text,
        "source": source,
        "topic": topic_label,
        "pmid": rec["pmid"],
        "journal": rec["journal"],
        "year": rec["year"],
    }


def main(out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen_pmids: set[str] = set()
    total_written = 0

    logger.info("Fetching PubMed abstracts for %d topic queries...", len(TOPICS))
    logger.info("Output: %s", out)
    logger.info("")

    with httpx.Client(timeout=30.0) as client, open(out, "w", encoding="utf-8") as f:
        for query, max_results, label in TOPICS:
            try:
                logger.info("Searching: %-30s (up to %d results)", label, max_results)

                pmids = search_pmids(query, max_results, client)
                new_pmids = [p for p in pmids if p not in seen_pmids]

                if not new_pmids:
                    logger.info("  -> No new results")
                    continue

                records = fetch_abstracts(new_pmids, client)

                written = 0
                for rec in records:
                    if rec["pmid"] in seen_pmids:
                        continue
                    seen_pmids.add(rec["pmid"])
                    jsonl_record = build_record(rec, label)
                    f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
                    written += 1
                    total_written += 1

                logger.info("  -> Wrote %d abstracts", written)

                # PubMed rate limit: 3 requests/sec without API key
                time.sleep(0.8)

            except httpx.HTTPError as e:
                logger.warning("HTTP error for topic '%s': %s — skipping", label, e)
                time.sleep(2.0)
                continue
            except Exception as e:
                logger.warning("Unexpected error for topic '%s': %s — skipping", label, e)
                continue

    logger.info("")
    logger.info("=" * 60)
    logger.info("PubMed fetch complete.")
    logger.info("   Total abstracts written : %d", total_written)
    logger.info("   Unique PMIDs            : %d", len(seen_pmids))
    logger.info("   Output file             : %s", out)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step — push to Pinecone:")
    logger.info("  python scripts/ingest.py --data %s --batch-size 16", out)
    logger.info("")
    logger.info("Note: ingesting ~%d vectors takes approx %d minutes on HF free tier.",
                total_written, max(1, total_written // 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch PubMed abstracts as MediQuery knowledge base"
    )
    parser.add_argument(
        "--out",
        default="data/pubmed_knowledge.jsonl",
        help="Output JSONL path (default: data/pubmed_knowledge.jsonl)",
    )
    args = parser.parse_args()
    main(args.out)