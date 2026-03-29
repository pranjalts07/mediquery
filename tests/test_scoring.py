"""
tests/test_scoring.py
Unit tests for the evaluation scoring functions in scripts/evaluate_mediquery.py.

Run with:
    pytest tests/test_scoring.py -v
"""
import math
import sys
from pathlib import Path

# The scripts/ directory is not a package — add it to the path so we can import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from evaluate_mediquery import (
    _cosine,
    _normalize,
    score_gt_overlap,
    score_keyword_recall,
    score_source_supported,
)


# ── _normalize ────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_strips_punctuation(self):
        assert "word" in _normalize("word.")
        assert "word" in _normalize("word,")

    def test_lowercases(self):
        assert "diabetes" in _normalize("Diabetes")

    def test_empty_string(self):
        assert _normalize("") == []

    def test_multiple_words(self):
        result = _normalize("Heart attack risk")
        assert "heart" in result
        assert "attack" in result
        assert "risk" in result


# ── _cosine ───────────────────────────────────────────────────────────────────

class TestCosine:
    def test_identical_vectors_score_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_score_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine(a, b)) < 1e-9

    def test_opposite_vectors_score_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine(a, b) + 1.0) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ── score_keyword_recall ──────────────────────────────────────────────────────

class TestKeywordRecall:
    def test_all_keywords_present(self):
        answer = "insulin is produced by the pancreas and regulates glucose"
        keywords = ["insulin", "pancreas", "glucose"]
        assert score_keyword_recall(answer, keywords) == 1.0

    def test_no_keywords_present(self):
        answer = "the weather is nice today"
        keywords = ["insulin", "pancreas", "glucose"]
        assert score_keyword_recall(answer, keywords) == 0.0

    def test_partial_keywords(self):
        answer = "insulin helps control glucose levels"
        keywords = ["insulin", "pancreas", "glucose"]
        score = score_keyword_recall(answer, keywords)
        assert abs(score - round(2 / 3, 3)) < 1e-9

    def test_empty_keywords_returns_zero(self):
        assert score_keyword_recall("any answer", []) == 0.0

    def test_case_insensitive(self):
        answer = "Hypertension is high blood pressure"
        keywords = ["hypertension", "BLOOD PRESSURE"]
        assert score_keyword_recall(answer, keywords) == 1.0


# ── score_gt_overlap ──────────────────────────────────────────────────────────

class TestGtOverlap:
    def test_identical_text_scores_one(self):
        text = "insulin pancreas glucose hyperglycemia"
        assert score_gt_overlap(text, text) == 1.0

    def test_no_overlap_scores_zero(self):
        answer = "completely unrelated response about weather"
        gt = "insulin pancreas glucose hyperglycemia"
        assert score_gt_overlap(answer, gt) == 0.0

    def test_short_words_excluded(self):
        # Words with 4 chars or fewer are excluded from overlap calculation
        answer = "the and for with"
        gt = "the and for with"
        # All words are ≤4 chars, so gt_words is empty → returns 0.0
        assert score_gt_overlap(answer, gt) == 0.0

    def test_partial_overlap(self):
        answer = "insulin is important for glucose regulation in the body"
        gt = "insulin controls blood glucose levels and prevents hyperglycemia"
        score = score_gt_overlap(answer, gt)
        assert 0.0 < score < 1.0

    def test_capped_at_one(self):
        # Answer repeats ground-truth words many times — should cap at 1.0
        answer = "glucose glucose glucose glucose glucose insulin insulin insulin"
        gt = "glucose insulin"
        assert score_gt_overlap(answer, gt) == 1.0


# ── score_source_supported ────────────────────────────────────────────────────

class TestSourceSupported:
    def test_no_sources_returns_zero(self):
        assert score_source_supported("any answer", []) == 0.0

    def test_answer_fully_in_sources(self):
        answer = "insulin controls glucose"
        sources = [{"text": "insulin controls glucose levels in the blood", "source": "MediQuery"}]
        score = score_source_supported(answer, sources)
        assert score > 0.5

    def test_answer_not_in_sources(self):
        answer = "completely fabricated information about dragons"
        sources = [{"text": "insulin is produced by the pancreas", "source": "MediQuery"}]
        score = score_source_supported(answer, sources)
        assert score < 0.3

    def test_empty_answer_returns_zero(self):
        sources = [{"text": "some source text", "source": "MediQuery"}]
        assert score_source_supported("", sources) == 0.0
