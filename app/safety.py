"""
app/safety.py
Lightweight keyword-based safety layer.
Only blocks genuine emergencies and prompt injection — not casual greetings.
"""
from __future__ import annotations
import re

EMERGENCY_KEYWORDS: list[str] = [
    "suicide", "kill myself", "end my life", "want to die",
    "self harm", "self-harm", "overdose", "od on",
    "heart attack", "stroke", "unconscious", "not breathing",
    "can't breathe", "cannot breathe", "severe bleeding",
    "call 911", "call 999", "call 112",
]

OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "hack", "exploit", "jailbreak",
    "ignore previous", "ignore instructions", "act as", "dan mode",
]

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

GREETING_PATTERNS: list[str] = [
    r"^h+e+y+[!.]*$", r"^h+i+[!.]*$", r"^hello[!.]*$", r"^howdy[!.]*$",
    r"^hiya[!.]*$", r"^yo[!.]*$", r"^sup[!.]*$",
    r"^how are you", r"^how r you", r"^whats up", r"^what'?s up",
    r"^good (morning|afternoon|evening|night)",
    r"^who are you", r"^what are you",
    r"^what can you do", r"^what do you do",
]

THANKS_PATTERNS: list[str] = [
    r"^(thanks|thank you|thank u|thx|ty|thankyou)[!.]*$",
]

BYE_PATTERNS: list[str] = [
    r"^(bye|goodbye|see you|cya|see ya)[!.]*$",
]


def _get_casual_response(cleaned: str) -> str:
    for p in THANKS_PATTERNS:
        if re.match(p, cleaned, re.IGNORECASE):
            return "You're welcome! Feel free to ask anything else about health or medical topics."
    for p in BYE_PATTERNS:
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
    for p in GREETING_PATTERNS:
        if re.match(p, cleaned, re.IGNORECASE):
            return "Hey there! I'm MediQuery, your AI medical assistant. What health question can I help you with today?"
    return None


class SafetyResult:
    __slots__ = ("blocked", "response")
    def __init__(self, blocked: bool, response: str = ""):
        self.blocked = blocked
        self.response = response


def check(text: str) -> SafetyResult:
    lowered = text.lower().strip()
    cleaned = re.sub(r'[!?.,:]+$', '', lowered).strip()

    for kw in EMERGENCY_KEYWORDS:
        if kw in lowered:
            return SafetyResult(blocked=True, response=EMERGENCY_RESPONSE)

    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in lowered:
            return SafetyResult(blocked=True, response=REFUSAL_RESPONSE)

    if len(cleaned.split()) <= 6:
        casual_response = _get_casual_response(cleaned)
        if casual_response:
            return SafetyResult(blocked=True, response=casual_response)

    return SafetyResult(blocked=False)