import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.metrics import metrics
from app.middleware import RequestIDFilter, request_id_middleware
from app.rag import _get_pinecone_index, get_circuit_breaker_states, run_rag, run_rag_stream
from app.safety import check as safety_check


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Global daily cap — prevents HF API quota exhaustion from distributed abuse.
# Resets every 24 hours. Configurable via DAILY_REQUEST_LIMIT env var.
_daily_lock = __import__("threading").Lock()
_daily_count = 0
_daily_reset_at = time.time() + 86400
DAILY_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1000"))
MAX_BODY_BYTES = 32 * 1024  # 32KB — well above any valid request


def _allow_request() -> bool:
    global _daily_count, _daily_reset_at
    with _daily_lock:
        if time.time() >= _daily_reset_at:
            _daily_count = 0
            _daily_reset_at = time.time() + 86400
        if _daily_count >= DAILY_LIMIT:
            return False
        _daily_count += 1
        return True


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "req": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log)


_use_json_logs = os.getenv("LOG_FORMAT", "json").lower() != "text"
_formatter = _JSONFormatter() if _use_json_logs else logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | req=%(request_id)s | %(message)s"
)
_request_id_filter = RequestIDFilter()

logging.basicConfig(level=logging.INFO)
for _handler in logging.root.handlers:
    _handler.setFormatter(_formatter)
    _handler.addFilter(_request_id_filter)

logger = logging.getLogger("mediquery")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

limiter = Limiter(key_func=_get_client_ip)

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("MediQuery started. Pinecone index: %s", settings.pinecone_index_name)
    try:
        await asyncio.to_thread(_get_pinecone_index, settings.pinecone_api_key, settings.pinecone_host)
        logger.info("Pinecone connection warmed up")
    except Exception as exc:
        logger.warning("Pinecone warmup failed (will retry on first request): %s", exc)
    yield


app = FastAPI(
    title="MediQuery",
    description="Medical RAG chatbot powered by Pinecone vector retrieval and HuggingFace LLM inference.",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"error": "Request body too large."})
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    return await request_id_middleware(request, call_next)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
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


# Reject null bytes and other non-printable control characters.
# Keeps \t (0x09), \n (0x0a), \r (0x0d) — legitimate in multi-line text.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant"})


def _sanitize_text(value: str) -> str:
    value = value.strip()
    value = unicodedata.normalize("NFC", value)
    if _CONTROL_CHAR_RE.search(value):
        raise ValueError("Message contains invalid control characters.")
    value = re.sub(r"[ \t]+", " ", value)
    return value


class HistoryMessage(BaseModel):
    model_config = {"extra": "forbid"}

    role: str = Field(..., min_length=1, max_length=20)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of: {sorted(_ALLOWED_ROLES)}")
        return v

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return _sanitize_text(v)


class ChatRequest(BaseModel):
    model_config = {"extra": "forbid"}

    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[HistoryMessage]] = Field(default=None, max_length=20)
    mode: Literal["simple", "short", "detailed"] = Field(default="detailed")
    patient_context: Optional[str] = Field(default=None, max_length=500)

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


class ReadinessResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@limiter.limit("60/minute")
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health(request: Request):
    return HealthResponse(
        status="ok",
        version="3.1.0",
        model=settings.hf_llm_model,
        index=settings.pinecone_index_name,
    )


@app.get("/health/ready", response_model=ReadinessResponse)
@limiter.limit("30/minute")
async def health_ready(request: Request):
    states = get_circuit_breaker_states()
    degraded = any(s != "closed" for s in states.values())
    return ReadinessResponse(
        status="degraded" if degraded else "ready",
        version="3.1.0",
        services=states,
    )


@app.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(request: Request):
    return JSONResponse(content={
        **metrics.summary(),
        "circuit_breakers": get_circuit_breaker_states(),
        "daily_quota": {
            "used": _daily_count,
            "limit": DAILY_LIMIT,
            "remaining": max(0, DAILY_LIMIT - _daily_count),
        },
    })


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    if not _allow_request():
        raise HTTPException(status_code=429, detail="Daily request limit reached. Try again tomorrow.")

    message = body.message

    safety_result = safety_check(message)
    if safety_result.blocked:
        logger.info("Safety block triggered (message length: %d)", len(message))
        return ChatResponse(answer=safety_result.response, sources=[])

    t0 = time.perf_counter()
    try:
        history = (
            [{"role": m.role, "content": m.content} for m in body.history]
            if body.history
            else None
        )
        result = await run_rag(message, settings, history=history, mode=body.mode, patient_context=body.patient_context)
    except RuntimeError as e:
        logger.error("RAG pipeline error: %s", e)
        metrics.record_request((time.perf_counter() - t0) * 1000, error=True)
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again shortly.",
        ) from e
    except Exception:
        logger.exception("Unexpected error in RAG pipeline")
        metrics.record_request((time.perf_counter() - t0) * 1000, error=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("RAG completed in %.2fs", elapsed)

    return ChatResponse(answer=result["answer"], sources=result.get("sources", []))


@app.post("/chat/stream", include_in_schema=False)
@limiter.limit("10/minute")
async def chat_stream(request: Request, body: ChatRequest):
    if not _allow_request():
        raise HTTPException(status_code=429, detail="Daily request limit reached. Try again tomorrow.")

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
        run_rag_stream(message, settings, history=history, mode=body.mode, patient_context=body.patient_context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred.", "answer": "", "sources": []},
    )
