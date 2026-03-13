"""
app/main.py
MediQuery FastAPI application entry point.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag import run_rag
from app.safety import check as safety_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mediquery")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="MediQuery",
    description="Medical RAG chatbot powered by Pinecone vector retrieval and HuggingFace LLM inference.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

settings = get_settings()
logger.info("MediQuery started. Pinecone index: %s", settings.pinecone_index_name)


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[HistoryMessage]] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    index: str


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="3.0.0",
        model=settings.hf_llm_model,
        index=settings.pinecone_index_name,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    message = body.message.strip()

    safety_result = safety_check(message)
    if safety_result.blocked:
        logger.info("Safety block triggered for message: %s", message[:80])
        return ChatResponse(answer=safety_result.response, sources=[])

    t0 = time.perf_counter()
    try:
        history = [{"role": m.role, "content": m.content} for m in body.history] if body.history else None
        result = run_rag(message, settings, history=history)
    except RuntimeError as e:
        logger.error("RAG pipeline error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again shortly.",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in RAG pipeline")
        raise HTTPException(status_code=500, detail="Internal server error.") from e

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("RAG completed in %.2fs", elapsed)

    return ChatResponse(answer=result["answer"], sources=result.get("sources", []))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred.", "answer": "", "sources": []},
    )