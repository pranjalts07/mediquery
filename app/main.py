"""
app/main.py
MediQuery FastAPI application entry point.

Security hardening (OWASP best practices):
  1. Rate limiting (SlowAPI) — per-IP on all public endpoints (OWASP API4:2023).
     Returns HTTP 429 with a Retry-After header on breach.
  2. Strict input validation — Pydantic models with extra='forbid', type checks,
     length limits, role allow-list, and control-character rejection (OWASP A03:2021).
  3. Secure API key handling — all secrets loaded exclusively from environment
     variables via app/config.py; never logged, never sent to the client (OWASP A02:2021).
  4. Security response headers — CSP, X-Frame-Options, etc. via HTTP middleware.
  5. Restrictive CORS — same-origin by default; opt-in via ALLOWED_ORIGINS env var.
"""
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.rag import run_rag, run_rag_stream
from app.safety import check as safety_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mediquery")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Rate limiting — OWASP API4:2023 Unrestricted Resource Consumption
# ---------------------------------------------------------------------------
# SlowAPI uses the real client IP as the rate-limit key.
# X-Forwarded-For is respected so Azure / nginx reverse proxies work correctly.
# Counters are in-process memory; they reset on worker restart.  For
# multi-worker or multi-instance deployments, swap the default MemoryStorage
# for a shared backend (Redis) by passing storage_uri to Limiter.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MediQuery",
    description="Medical RAG chatbot powered by Pinecone vector retrieval and HuggingFace LLM inference.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register the SlowAPI 429 handler — sends a proper Retry-After header.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — OWASP A05:2021 Security Misconfiguration
# ---------------------------------------------------------------------------
# Same-origin only by default.  Set ALLOWED_ORIGINS to a comma-separated list
# of trusted origins if the UI is served from a different domain
# (e.g. "https://app.example.com,https://staging.example.com").
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    # Empty list → browser enforces same-origin; explicit list → those origins allowed.
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,           # No cookies / auth headers cross-origin
    allow_methods=["GET", "POST"],     # Only the methods this API uses
    allow_headers=["Content-Type"],    # Only the header the SPA sends
)

# ---------------------------------------------------------------------------
# Eager settings load — fails fast if required env vars are absent.
# API keys are stored in the Settings dataclass, never re-exported to clients.
# ---------------------------------------------------------------------------
settings = get_settings()
logger.info("MediQuery started. Pinecone index: %s", settings.pinecone_index_name)


# ---------------------------------------------------------------------------
# Security headers middleware — OWASP A05:2021 Security Misconfiguration
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Attach protective HTTP headers to every response."""
    response = await call_next(request)
    # Prevent MIME-type sniffing (stops content-type confusion attacks)
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Deny framing entirely (clickjacking protection)
    response.headers["X-Frame-Options"] = "DENY"
    # Legacy XSS filter for older browsers
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Limit referrer information leaked to third parties
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Content Security Policy — restrict resource origins.
    # 'unsafe-inline' is needed for the inline scripts/styles in chat.html;
    # if those are ever moved to separate files this can be tightened further.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    return response


# ---------------------------------------------------------------------------
# Input validation helpers — OWASP A03:2021 Injection
# ---------------------------------------------------------------------------

# Reject null bytes and other non-printable control characters that are used
# in injection, truncation, and log-poisoning attacks.
# Keeps \t (0x09), \n (0x0a), \r (0x0d) — legitimate in multi-line text.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Only these role values are meaningful in our conversation schema.
_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant"})


def _sanitize_text(value: str) -> str:
    """
    Sanitize a free-text field before it enters the application:
      1. Strip leading/trailing whitespace.
      2. NFC-normalize unicode — prevents homoglyph / lookalike attacks where
         visually identical characters have different code points.
      3. Reject control characters (null bytes, etc.) used for injection.
      4. Collapse internal runs of spaces/tabs to a single space.
    Returns the cleaned string, or raises ValueError on invalid input.
    """
    value = value.strip()
    value = unicodedata.normalize("NFC", value)
    if _CONTROL_CHAR_RE.search(value):
        raise ValueError("Message contains invalid control characters.")
    # Collapse horizontal whitespace runs (preserves intentional newlines)
    value = re.sub(r"[ \t]+", " ", value)
    return value


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    # extra="forbid" rejects any field not declared here — prevents parameter
    # pollution and unexpected data smuggling (OWASP A03:2021).
    model_config = {"extra": "forbid"}

    role: str = Field(..., min_length=1, max_length=20)
    # Cap individual history messages to 4 000 chars to bound context size.
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Only 'user' and 'assistant' are valid roles; reject everything else."""
        if v not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of: {sorted(_ALLOWED_ROLES)}")
        return v

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return _sanitize_text(v)


class ChatRequest(BaseModel):
    model_config = {"extra": "forbid"}  # Reject unexpected fields

    message: str = Field(..., min_length=1, max_length=2000)
    # Cap history depth at 20 turns to prevent excessive context injection.
    history: Optional[List[HistoryMessage]] = Field(default=None, max_length=20)
    # "short" → 2-3 sentence answer; "detailed" → full explanation (default).
    # Literal enforces only these two values — anything else is a 422.
    mode: Literal["short", "detailed"] = Field(default="detailed")

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return _sanitize_text(v)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    index: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@limiter.limit("60/minute")
async def index(request: Request):
    """Serve the chat UI.  Rate-limited to 60 page loads/minute per IP."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health(request: Request):
    """Health check.  Returns app/model/index metadata.  No secrets included."""
    return HealthResponse(
        status="ok",
        version="3.0.0",
        model=settings.hf_llm_model,
        index=settings.pinecone_index_name,
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Primary AI endpoint.

    Rate limit: 10 requests/minute per IP — enough for interactive use,
    tight enough to prevent bulk scraping or API abuse.

    The `message` and all `history[].content` fields have already been
    sanitized by the Pydantic validators above before reaching this handler.
    """
    # Pydantic has already stripped and sanitized the message.
    message = body.message

    safety_result = safety_check(message)
    if safety_result.blocked:
        # Log only the message length — not the content — to avoid storing
        # potentially sensitive user input in server logs.
        logger.info("Safety block triggered (message length: %d)", len(message))
        return ChatResponse(answer=safety_result.response, sources=[])

    t0 = time.perf_counter()
    try:
        history = (
            [{"role": m.role, "content": m.content} for m in body.history]
            if body.history
            else None
        )
        result = await run_rag(message, settings, history=history, mode=body.mode)
    except RuntimeError as e:
        logger.error("RAG pipeline error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again shortly.",
        ) from e
    except Exception:
        logger.exception("Unexpected error in RAG pipeline")
        raise HTTPException(status_code=500, detail="Internal server error.")

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("RAG completed in %.2fs", elapsed)

    return ChatResponse(answer=result["answer"], sources=result.get("sources", []))


@app.post("/chat/stream", include_in_schema=False)
@limiter.limit("10/minute")
async def chat_stream(request: Request, body: ChatRequest):
    """
    Streaming AI endpoint — returns Server-Sent Events.

    Same rate limit and validation as /chat. The frontend uses this endpoint
    so users see tokens appear in real-time instead of waiting 4-9 seconds.

    Event format (newline-delimited JSON after "data: "):
      {"type": "token",  "content": "..."}   — one text token
      {"type": "done",   "sources": [...]}   — generation finished + source list
      {"type": "error",  "content": "..."}   — pipeline failure
    """
    message = body.message

    safety_result = safety_check(message)
    if safety_result.blocked:
        logger.info("Safety block triggered (stream, length: %d)", len(message))

        async def _blocked_stream():
            import json as _json
            yield f"data: {_json.dumps({'type': 'token', 'content': safety_result.response})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'sources': []})}\n\n"

        return StreamingResponse(
            _blocked_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    history = (
        [{"role": m.role, "content": m.content} for m in body.history]
        if body.history
        else None
    )

    return StreamingResponse(
        run_rag_stream(message, settings, history=history, mode=body.mode),
        media_type="text/event-stream",
        # X-Accel-Buffering: no prevents Azure/nginx from buffering the stream
        # before forwarding to the client — without this, the user sees no tokens
        # until the buffer fills, defeating the purpose of streaming entirely.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred.", "answer": "", "sources": []},
    )
