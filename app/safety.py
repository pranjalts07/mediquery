"""
app/safety.py
Lightweight keyword-based safety layer.

Two-tier emergency detection:
  1. Crisis keywords — suicide/self-harm terms are always high-risk regardless
     of how they appear in a sentence.  False negative (missing a real crisis)
     is far more dangerous than a false positive here.

  2. Acute medical phrases — "heart attack" and "stroke" are core medical topics
     that the chatbot must be able to discuss.  Blocking any message that contains
     these words would make a medical assistant useless ("what causes heart attacks?"
     would be refused).  Instead, we only block when the message also contains
     first-person present-tense emergency indicators ("I'm having", "I can't breathe").
"""
from __future__ import annotations
import re

# ── Tier 1: always-block crisis keywords ─────────────────────────────────────
# Mental health crises and direct emergency calls.
# The risk of missing a genuine crisis outweighs occasional false positives.
_CRISIS_KEYWORDS: frozenset[str] = frozenset({
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "self harm",
    "self-harm",
    "overdose",
    "od on",
    "severe bleeding",
    "call 911",
    "call 999",
    "call 112",
})

# ── Tier 2: medical emergency terms + acute indicators ───────────────────────
# Only block when BOTH a medical emergency term AND a first-person acute
# indicator appear in the same message.
# "what causes heart attacks?" → no acute indicator → passes through to RAG
# "I think I'm having a heart attack" → blocked immediately
_MEDICAL_EMERGENCY_TERMS: frozenset[str] = frozenset({
    "heart attack",
    "stroke",
    "not breathing",
    "stopped breathing",
    "cant breathe",
    "unconscious",
    "passed out",
    "chest pain",
})

_ACUTE_INDICATORS: tuple[str, ...] = (
    "i'm having",
    "i am having",
    "im having",
    "i think i'm",
    "i think im",
    "i think i am",
    "i can't breathe",
    "i cannot breathe",
    "i'm not breathing",
    "im not breathing",
    "right now",
    "happening now",
    "please help",
    "help me",
    "emergency",
    "someone is having",
    "they are having",
    "they're having",
)

# ── Prompt injection / out-of-scope ──────────────────────────────────────────
_OUT_OF_SCOPE_KEYWORDS: tuple[str, ...] = (
    "hack",
    "exploit",
    "jailbreak",
    "ignore previous",
    "ignore instructions",
    "act as",
    "dan mode",
)

# ── Canned responses ─────────────────────────────────────────────────────────
EMERGENCY_RESPONSE = (
    "⚠️ **This sounds like a medical emergency.** "
    "Please call your local emergency number immediately "
    "(911 in the US · 999 in the UK · 112 in the EU). "
    "Do not wait — go to your nearest emergency room or call an ambulance now.\n\n"
    "*MediQuery provides general health information only and cannot replace "
    "emergency medical services.*"
)

REFUSAL_RESPONSE = (
    "I'm sorry, I can only answer general health and medical questions. "
    "I'm not able to help with that request."
)

# ── Casual-response patterns ──────────────────────────────────────────────────
_GREETING_PATTERNS: tuple[str, ...] = (
    r"^h+e+y+[!.]*$", r"^h+i+[!.]*$", r"^hello[!.]*$", r"^howdy[!.]*$",
    r"^hiya[!.]*$", r"^yo[!.]*$", r"^sup[!.]*$",
    r"^how are you", r"^how r you", r"^whats up", r"^what'?s up",
    r"^good (morning|afternoon|evening|night)",
    r"^who are you", r"^what are you",
    r"^what can you do", r"^what do you do",
)

_THANKS_PATTERNS: tuple[str, ...] = (
    r"^(thanks|thank you|thank u|thx|ty|thankyou)[!.]*$",
)

_BYE_PATTERNS: tuple[str, ...] = (
    r"^(bye|goodbye|see you|cya|see ya)[!.]*$",
)


def _get_casual_response(cleaned: str) -> str | None:
    for p in _THANKS_PATTERNS:
        if re.match(p, cleaned, re.IGNORECASE):
            return "You're welcome! Feel free to ask anything else about health or medical topics."
    for p in _BYE_PATTERNS:
        if re.match(p, cleaned, re.IGNORECASE):
            return "Take care! Come back anytime you have health questions."
    if cleaned in ("help", "help!"):
        return (
            "I can help you with:\n\n"
            "• **Symptoms** — understanding what they might mean\n"
            "• **Conditions** — learning about diseases and disorders\n"
            "• **Medications** — how drugs work and their uses\n"
            "• **General health** — wellness, prevention, and more\n\n"
            "Just type your question and I'll do my best to help!"
        )
    for p in _GREETING_PATTERNS:
        if re.match(p, cleaned, re.IGNORECASE):
            return "Hey there! I'm MediQuery, your AI medical assistant. What health question can I help you with today?"
    return None


# ── Public interface ──────────────────────────────────────────────────────────

class SafetyResult:
    __slots__ = ("blocked", "response")

    def __init__(self, blocked: bool, response: str = ""):
        self.blocked = blocked
        self.response = response


def check(text: str) -> SafetyResult:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[!?.,:]+$", "", lowered).strip()

    # Tier 1 — crisis keywords (always block)
    for kw in _CRISIS_KEYWORDS:
        if kw in lowered:
            return SafetyResult(blocked=True, response=EMERGENCY_RESPONSE)

    # Tier 2 — medical emergency terms need an acute first-person indicator
    has_medical_term = any(term in lowered for term in _MEDICAL_EMERGENCY_TERMS)
    has_acute = any(phrase in lowered for phrase in _ACUTE_INDICATORS)
    if has_medical_term and has_acute:
        return SafetyResult(blocked=True, response=EMERGENCY_RESPONSE)

    # Out-of-scope / prompt injection
    for kw in _OUT_OF_SCOPE_KEYWORDS:
        if kw in lowered:
            return SafetyResult(blocked=True, response=REFUSAL_RESPONSE)

    # Casual conversational messages (short messages only)
    if len(cleaned.split()) <= 6:
        casual = _get_casual_response(cleaned)
        if casual:
            return SafetyResult(blocked=True, response=casual)

    return SafetyResult(blocked=False)
