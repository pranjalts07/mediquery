"""scripts/evaluate_mediquery.py — End-to-end quality evaluation for MediQuery.

Usage:
  python scripts/evaluate_mediquery.py
  python scripts/evaluate_mediquery.py --url https://staging.example.com --mode short --no-semantic
  python scripts/evaluate_mediquery.py --categories cardiovascular diabetes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = os.getenv("MEDIQUERY_URL", "http://localhost:8000")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_URL = (
    f"https://router.huggingface.co/hf-inference/models/"
    f"{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"
)

TIMEOUT = 90
REQUEST_DELAY_SECONDS = 4
RETRIES = 3

TEST_CASES = [
    # cardiovascular
    {
        "category": "cardiovascular",
        "question": "my doctor said i have high blood pressure, what does that actually mean and why is it bad?",
        "ground_truth": "Hypertension is persistently elevated blood pressure above 130/80 mmHg. Risk factors include obesity, high sodium diet, physical inactivity, smoking, alcohol, family history, and stress. It increases risk of heart attack, stroke, and kidney disease.",
        "keywords": ["hypertension", "blood pressure", "heart attack", "stroke", "kidney disease", "sodium"],
    },
    {
        "category": "cardiovascular",
        "question": "whats the difference between the two numbers they always measure in blood pressure",
        "ground_truth": "Systolic blood pressure is the pressure when the heart beats and pumps blood. Diastolic is the pressure when the heart rests between beats. Normal is below 120/80 mmHg.",
        "keywords": ["systolic", "diastolic", "blood pressure", "heart", "120/80", "mmHg"],
    },
    {
        "category": "cardiovascular",
        "question": "i keep hearing about bad cholesterol, can you explain what ldl actually does",
        "ground_truth": "LDL cholesterol carries fat from the liver to cells. High LDL causes plaque buildup in artery walls leading to atherosclerosis, increasing risk of heart attack and stroke.",
        "keywords": ["LDL", "cholesterol", "plaque", "artery", "atherosclerosis", "stroke"],
    },
    {
        "category": "cardiovascular",
        "question": "my dad had a heart attack last year, how do i know if im at risk",
        "ground_truth": "Heart attack risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, family history, and physical inactivity. A doctor can assess personal risk.",
        "keywords": ["heart attack", "risk", "blood pressure", "cholesterol", "smoking", "family history", "diabetes"],
    },
    # diabetes
    {
        "category": "diabetes",
        "question": "im always thirsty and peeing a lot, could that be diabetes",
        "ground_truth": "Frequent urination and excessive thirst are classic symptoms of diabetes. Other symptoms include fatigue, blurred vision, slow-healing sores, and unexplained weight loss.",
        "keywords": ["diabetes", "thirst", "urination", "fatigue", "blurred vision", "weight loss"],
    },
    {
        "category": "diabetes",
        "question": "how does insulin actually work in the body",
        "ground_truth": "Insulin is a hormone produced by the pancreas that allows cells to absorb glucose from the blood for energy. Without insulin, glucose builds up in the bloodstream causing hyperglycemia.",
        "keywords": ["insulin", "pancreas", "glucose", "blood", "cells", "hyperglycemia"],
    },
    {
        "category": "diabetes",
        "question": "whats the difference between type 1 and type 2 diabetes",
        "ground_truth": "Type 1 diabetes is autoimmune where the body destroys insulin-producing beta cells. Type 2 is where cells become resistant to insulin, often linked to obesity and lifestyle.",
        "keywords": ["type 1", "type 2", "autoimmune", "insulin", "resistant", "beta cells"],
    },
    # respiratory
    {
        "category": "respiratory",
        "question": "i get really short of breath when i exercise, could it be asthma",
        "ground_truth": "Exercise-induced asthma causes airway narrowing during physical activity. Symptoms include shortness of breath, wheezing, and chest tightness. Treatment includes bronchodilator inhalers.",
        "keywords": ["asthma", "exercise", "airway", "wheezing", "chest tightness", "inhaler"],
    },
    {
        "category": "respiratory",
        "question": "how does smoking actually damage your lungs",
        "ground_truth": "Smoking causes chronic inflammation, destroys air sacs reducing lung capacity, leads to COPD and emphysema, paralyses cilia so mucus builds up, and significantly raises lung cancer risk.",
        "keywords": ["smoking", "lungs", "inflammation", "air sacs", "COPD", "emphysema", "lung cancer", "cilia"],
    },
    # infectious
    {
        "category": "infectious",
        "question": "whats the difference between a cold and the flu",
        "ground_truth": "Colds have gradual onset and mainly affect the nose and throat. Flu has sudden onset with fever, body aches, and fatigue and can cause serious complications.",
        "keywords": ["cold", "flu", "fever", "body aches", "fatigue", "gradual", "sudden"],
    },
    # musculoskeletal
    {
        "category": "musculoskeletal",
        "question": "my joints are always swollen and painful in the morning, what could cause that",
        "ground_truth": "Morning joint stiffness and swelling is a hallmark of rheumatoid arthritis, an autoimmune disease. It can also indicate gout or other inflammatory conditions.",
        "keywords": ["rheumatoid arthritis", "joint", "swelling", "morning stiffness", "autoimmune", "inflammatory"],
    },
    {
        "category": "musculoskeletal",
        "question": "whats the difference between osteoarthritis and rheumatoid arthritis",
        "ground_truth": "Osteoarthritis is mechanical wear-and-tear joint damage. Rheumatoid arthritis is autoimmune where the immune system attacks the joint lining causing systemic inflammation.",
        "keywords": ["osteoarthritis", "rheumatoid arthritis", "wear-and-tear", "autoimmune", "joint lining", "inflammation"],
    },
    # metabolic
    {
        "category": "metabolic",
        "question": "besides looking overweight, what are the actual health problems obesity causes",
        "ground_truth": "Obesity increases risk of type 2 diabetes, cardiovascular disease, hypertension, sleep apnea, certain cancers, osteoarthritis, and fatty liver disease.",
        "keywords": ["obesity", "type 2 diabetes", "cardiovascular disease", "hypertension", "sleep apnea", "cancer", "fatty liver"],
    },
    {
        "category": "metabolic",
        "question": "what is BMI and is it actually a good measure of health",
        "ground_truth": "BMI is body mass index calculated from height and weight. It is a screening tool but has limitations as it does not account for muscle mass or fat distribution.",
        "keywords": ["BMI", "body mass index", "height", "weight", "screening tool", "muscle mass", "fat distribution"],
    },
    # immunology
    {
        "category": "immunology",
        "question": "how does my immune system actually fight off an infection",
        "ground_truth": "The immune system uses neutrophils and macrophages as first responders and B cells producing antibodies for targeted defence. Phagocytes engulf and destroy pathogens.",
        "keywords": ["immune system", "infection", "neutrophils", "macrophages", "B cells", "antibodies", "pathogens"],
    },
    {
        "category": "immunology",
        "question": "why do i get a fever when im sick",
        "ground_truth": "Fever is a controlled immune response where the hypothalamus raises body temperature to inhibit bacterial and viral growth and speed up immune cell activity.",
        "keywords": ["fever", "immune response", "hypothalamus", "temperature", "bacterial", "viral", "immune cells"],
    },
    {
        "category": "immunology",
        "question": "whats an autoimmune disease and why does the body attack itself",
        "ground_truth": "Autoimmune diseases occur when the immune system mistakenly attacks healthy cells. The cause involves a mix of genetic predisposition and environmental triggers.",
        "keywords": ["autoimmune", "immune system", "healthy cells", "genetic", "environmental triggers", "attacks itself"],
    },
    # mental health
    {
        "category": "mental_health",
        "question": "whats the difference between feeling sad and actually having depression",
        "ground_truth": "Depression is a clinical condition involving persistent low mood, loss of interest, fatigue, sleep changes, and cognitive difficulties lasting more than two weeks.",
        "keywords": ["depression", "persistent", "low mood", "loss of interest", "fatigue", "sleep changes", "two weeks"],
    },
    {
        "category": "mental_health",
        "question": "can stress actually make you physically sick",
        "ground_truth": "Chronic stress raises cortisol levels which suppresses immune function, raises blood pressure, disrupts sleep, and increases risk of cardiovascular disease.",
        "keywords": ["stress", "cortisol", "immune function", "blood pressure", "sleep", "cardiovascular disease", "chronic"],
    },
    # nutrition
    {
        "category": "nutrition",
        "question": "why is too much salt bad for you",
        "ground_truth": "Excess sodium causes the body to retain water, raising blood volume and blood pressure, which strains the heart and arteries and increases risk of hypertension.",
        "keywords": ["salt", "sodium", "retain water", "blood volume", "blood pressure", "heart", "hypertension"],
    },
    # organ function
    {
        "category": "organ_function",
        "question": "what does the liver actually do",
        "ground_truth": "The liver filters blood, produces bile for digestion, metabolises drugs and toxins, synthesises proteins, and stores glycogen for energy.",
        "keywords": ["liver", "filters blood", "bile", "digestion", "toxins", "proteins", "glycogen"],
    },
    {
        "category": "organ_function",
        "question": "what do the kidneys do and how do i know if they arent working properly",
        "ground_truth": "Kidneys filter waste from blood, regulate fluid balance, and control blood pressure. Signs of dysfunction include swelling, fatigue, changes in urination, and back pain.",
        "keywords": ["kidneys", "filter waste", "fluid balance", "blood pressure", "swelling", "fatigue", "urination"],
    },
    # oncology
    {
        "category": "oncology",
        "question": "what are the general warning signs that could indicate cancer",
        "ground_truth": "General cancer warning signs include unexplained weight loss, persistent fatigue, unusual lumps, changes in bowel habits, persistent cough, and unexplained bleeding.",
        "keywords": ["cancer", "weight loss", "fatigue", "lumps", "bowel habits", "cough", "bleeding"],
    },
    # endocrine
    {
        "category": "endocrine",
        "question": "ive been gaining weight and feeling tired all the time, could it be my thyroid",
        "ground_truth": "Hypothyroidism occurs when the thyroid produces too little hormone, causing weight gain, fatigue, cold intolerance, and depression.",
        "keywords": ["thyroid", "hypothyroidism", "weight gain", "fatigue", "hormone", "cold intolerance", "depression"],
    },
    # lifestyle
    {
        "category": "lifestyle",
        "question": "why is sleep so important for your health",
        "ground_truth": "Sleep is essential for immune function, memory consolidation, hormone regulation, tissue repair, and cardiovascular health. Chronic deprivation raises disease risk.",
        "keywords": ["sleep", "immune function", "memory", "hormone regulation", "tissue repair", "cardiovascular health", "deprivation"],
    },
]

ALL_CATEGORIES: list[str] = sorted({c["category"] for c in TEST_CASES})


def _normalize(text: str) -> list[str]:
    return [
        w.strip(".,!?;:()[]{}\"'")
        for w in text.lower().split()
        if w.strip(".,!?;:()[]{}\"'")
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def score_keyword_recall(answer: str, keywords: list[str]) -> float:
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return round(hits / len(keywords), 3) if keywords else 0.0


def score_gt_overlap(answer: str, ground_truth: str) -> float:
    gt_words = {w for w in _normalize(ground_truth) if len(w) > 4}
    ans_words = [w for w in _normalize(answer) if len(w) > 4]
    if not gt_words or not ans_words:
        return 0.0
    overlap = sum(1 for w in ans_words if w in gt_words)
    return round(min(overlap / len(gt_words), 1.0), 3)


def score_source_supported(answer: str, sources: list[dict]) -> float:
    if not sources:
        return 0.0
    source_text = " ".join(
        (s.get("text", "") + " " + s.get("source", "")) for s in sources
    ).lower()
    ans_words = [w for w in _normalize(answer) if len(w) > 4]
    if not ans_words:
        return 0.0
    supported = sum(1 for w in ans_words if w in source_text)
    return round(min(supported / len(ans_words), 1.0), 3)


def score_semantic_similarity(answer: str, ground_truth: str) -> float | None:
    """Cosine similarity of answer vs ground-truth embeddings. Returns None if unavailable."""
    if not HF_API_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [answer[:512], ground_truth[:512]],
        "options": {"wait_for_model": True},
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(EMBED_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            return None
        vecs = resp.json()
        if isinstance(vecs, list) and len(vecs) == 2 and isinstance(vecs[0], list):
            return round(_cosine(vecs[0], vecs[1]), 3)
    except Exception:
        pass
    return None


def query_mediquery(question: str, endpoint: str, mode: str = "detailed") -> dict:
    for attempt in range(RETRIES):
        try:
            t0 = time.time()
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(endpoint, json={"message": question, "mode": mode})
            resp.raise_for_status()
            data = resp.json()
            data["latency_ms"] = round((time.time() - t0) * 1000, 1)
            return data
        except Exception as exc:
            if attempt == RETRIES - 1:
                raise
            delay = 3 + random.random()
            print(f"  retry {attempt + 2}/{RETRIES} in {delay:.1f}s...", end=" ")
            time.sleep(delay)


def grade(score: float) -> str:
    if score >= 0.80:
        return "A"
    if score >= 0.65:
        return "B"
    if score >= 0.50:
        return "C"
    return "D"


def _overall(scores: dict) -> float:
    vals = [
        scores["keyword_recall"],
        scores["ground_truth_overlap"],
        scores["source_supported"],
    ]
    if scores.get("semantic_similarity") is not None:
        vals.append(scores["semantic_similarity"])
    return round(statistics.mean(vals), 3)


def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def _stdev_str(vals: list[float]) -> str:
    return f" ±{statistics.stdev(vals):.3f}" if len(vals) > 1 else ""


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("─" * max(len(title), 52))


def print_scorecard(results: list[dict], latencies: list[float]) -> float:
    kw  = [r["scores"]["keyword_recall"] for r in results]
    gt  = [r["scores"]["ground_truth_overlap"] for r in results]
    src = [r["scores"]["source_supported"] for r in results]
    sem = [r["scores"]["semantic_similarity"] for r in results
           if r["scores"]["semantic_similarity"] is not None]

    avg_kw  = statistics.mean(kw)
    avg_gt  = statistics.mean(gt)
    avg_src = statistics.mean(src)
    avg_sem = statistics.mean(sem) if sem else None

    overall_vals = [avg_kw, avg_gt, avg_src]
    if avg_sem is not None:
        overall_vals.append(avg_sem)
    overall = round(statistics.mean(overall_vals), 3)

    p95 = sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    w = 24

    _print_section("Quality metrics")
    print(f"  {'Keyword recall':{w}}  {_pct(avg_kw)}{_stdev_str(kw)}")
    print(f"  {'Ground-truth overlap':{w}}  {_pct(avg_gt)}{_stdev_str(gt)}")
    print(f"  {'Source-supported':{w}}  {_pct(avg_src)}{_stdev_str(src)}")
    if avg_sem is not None:
        print(f"  {'Semantic similarity':{w}}  {_pct(avg_sem)}{_stdev_str(sem)}")
    print(f"  {'─' * (w + 14)}")
    print(f"  {'Overall':{w}}  {overall:.3f}   grade: {grade(overall)}")

    _print_section("Latency")
    print(f"  {'Median':{w}}  {statistics.median(latencies):.0f} ms")
    print(f"  {'Mean':{w}}  {statistics.mean(latencies):.0f} ms")
    print(f"  {'P95':{w}}  {p95:.0f} ms")
    print(f"  {'Min':{w}}  {min(latencies):.0f} ms")

    return overall


def print_category_breakdown(results: list[dict]) -> None:
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "unknown"), []).append(r)

    _print_section("Per-category breakdown")
    col = 20
    header = f"  {'Category':<{col}} {'N':>3}  {'KW%':>6}  {'GT%':>6}  {'Src%':>6}  {'Sem%':>6}  {'Avg%':>6}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for cat in sorted(by_cat):
        rs = by_cat[cat]
        kw  = statistics.mean(r["scores"]["keyword_recall"] for r in rs)
        gt  = statistics.mean(r["scores"]["ground_truth_overlap"] for r in rs)
        src = statistics.mean(r["scores"]["source_supported"] for r in rs)
        sems = [r["scores"]["semantic_similarity"] for r in rs
                if r["scores"]["semantic_similarity"] is not None]
        sem_avg = statistics.mean(sems) if sems else None
        sem_str = f"{sem_avg * 100:5.1f}%" if sem_avg is not None else "  n/a "
        avg_vals = [kw, gt, src] + ([sem_avg] if sem_avg is not None else [])
        avg = statistics.mean(avg_vals)
        print(f"  {cat:<{col}} {len(rs):>3}  {kw*100:5.1f}%  {gt*100:5.1f}%  {src*100:5.1f}%  {sem_str}  {avg*100:5.1f}%")


def print_worst(results: list[dict], n: int = 5) -> None:
    ranked = sorted(results, key=lambda r: _overall(r["scores"]))[:n]
    _print_section(f"Bottom {n} — hardest questions for current retrieval")
    for i, r in enumerate(ranked, 1):
        avg = _overall(r["scores"])
        s = r["scores"]
        sem_part = f"  sem={s['semantic_similarity']}" if s["semantic_similarity"] is not None else ""
        print(f"  {i}. [{avg:.2f}] {r['question'][:72]}")
        print(f"     kw={s['keyword_recall']}  gt={s['ground_truth_overlap']}  src={s['source_supported']}{sem_part}")
        print()


def run_evaluation(
    base_url: str,
    out_path: str | None,
    mode: str,
    use_semantic: bool,
    categories: list[str] | None,
) -> None:
    endpoint = f"{base_url.rstrip('/')}/chat"

    cases = TEST_CASES
    if categories:
        cases = [c for c in TEST_CASES if c["category"] in categories]
        if not cases:
            print(f"No test cases for categories: {categories}")
            return

    sem_enabled = use_semantic and bool(HF_API_TOKEN)

    print("MediQuery Evaluation")
    print(f"  target:    {base_url}")
    print(f"  mode:      {mode}")
    print(f"  questions: {len(cases)}")
    print(f"  semantic:  {'enabled' if sem_enabled else 'disabled (set HF_API_TOKEN to enable)'}")
    print(f"  started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results: list[dict] = []
    latencies: list[float] = []

    for i, case in enumerate(cases, 1):
        q = case["question"]
        print(f"  ({i:02d}/{len(cases)}) [{case['category']}] {q[:55]}...", end=" ", flush=True)

        try:
            response = query_mediquery(q, endpoint, mode=mode)
            answer   = response.get("answer", "")
            sources  = response.get("sources", [])

            kw_score  = score_keyword_recall(answer, case["keywords"])
            gt_score  = score_gt_overlap(answer, case["ground_truth"])
            src_score = score_source_supported(answer, sources)
            sem_score = score_semantic_similarity(answer, case["ground_truth"]) if sem_enabled else None

            result = {
                "question":      q,
                "category":      case["category"],
                "ground_truth":  case["ground_truth"],
                "answer":        answer,
                "latency_ms":    response["latency_ms"],
                "sources":       [{"source": s.get("source", ""), "score": s.get("score")} for s in sources],
                "scores": {
                    "keyword_recall":       kw_score,
                    "ground_truth_overlap": gt_score,
                    "source_supported":     src_score,
                    "semantic_similarity":  sem_score,
                },
            }
            results.append(result)
            latencies.append(response["latency_ms"])

            sem_tag = f"  sem={sem_score:.2f}" if sem_score is not None else ""
            print(f"{response['latency_ms']:.0f}ms  kw={kw_score:.2f}  gt={gt_score:.2f}{sem_tag}")

        except Exception as exc:
            print(f"FAILED: {exc}")

        time.sleep(REQUEST_DELAY_SECONDS)

    valid = [r for r in results if r is not None]
    success_rate = round(len(valid) / len(cases), 3) if cases else 0.0

    print(f"\n  {len(valid)}/{len(cases)} queries succeeded ({success_rate:.1%})")

    if not valid:
        print("No successful queries — cannot compute metrics.")
        return

    overall = print_scorecard(valid, latencies)
    print_category_breakdown(valid)
    print_worst(valid)

    kw  = [r["scores"]["keyword_recall"] for r in valid]
    gt  = [r["scores"]["ground_truth_overlap"] for r in valid]
    src = [r["scores"]["source_supported"] for r in valid]
    sem = [r["scores"]["semantic_similarity"] for r in valid
           if r["scores"]["semantic_similarity"] is not None]
    p95 = sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    output = {
        "timestamp":          datetime.now().isoformat(),
        "target_url":         base_url,
        "mode":               mode,
        "total_queries":      len(cases),
        "successful_queries": len(valid),
        "success_rate":       success_rate,
        "latency_stats": {
            "min_ms":    round(min(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "mean_ms":   round(statistics.mean(latencies), 1),
            "p95_ms":    round(p95, 1),
        },
        "quality_scores": {
            "keyword_recall":       round(statistics.mean(kw), 3),
            "ground_truth_overlap": round(statistics.mean(gt), 3),
            "source_supported":     round(statistics.mean(src), 3),
            "semantic_similarity":  round(statistics.mean(sem), 3) if sem else None,
        },
        "overall_score": overall,
        "grade":         grade(overall),
        "per_query":     valid,
    }

    fname = out_path or f"mediquery_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(fname).write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved -> {fname}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end quality evaluation for MediQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the MediQuery server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output JSON path (default: mediquery_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "detailed"],
        default="detailed",
        help="Response mode to evaluate (default: detailed)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic similarity scoring (saves ~1 HF API call per question)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        choices=ALL_CATEGORIES,
        help=f"Only evaluate these categories: {ALL_CATEGORIES}",
    )
    args = parser.parse_args()

    print(f"Checking server at {args.url}...")
    try:
        httpx.get(f"{args.url.rstrip('/')}/health", timeout=15)
        print("Server is up.\n")
    except Exception as exc:
        print(f"Cannot reach server: {exc}")
        raise SystemExit(1)

    run_evaluation(
        base_url=args.url,
        out_path=args.out,
        mode=args.mode,
        use_semantic=not args.no_semantic,
        categories=args.categories,
    )