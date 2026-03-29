"""
app/config.py
Centralised configuration loaded from environment variables.

Security contract (OWASP A02:2021 Cryptographic Failures / A05 Misconfiguration):
  - All secrets are injected at runtime — NEVER hardcoded in source code.
  - For local development, copy .env.example → .env and fill in values.
    The .env file is listed in .gitignore and must never be committed.
  - In production (Azure App Service), secrets are set as Application Settings,
    which are injected as environment variables and never appear in source or logs.
  - The Settings dataclass is frozen (immutable) to prevent accidental mutation.
  - __repr__ is overridden to ensure secrets cannot be leaked via print() or
    repr() calls in logs or error tracebacks.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Hugging Face
    hf_api_token: str
    hf_embedding_model: str
    hf_llm_model: str
    hf_reranker_model: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_host: str

    # App
    top_k: int
    max_new_tokens: int

    def __repr__(self) -> str:
        """
        Custom repr that masks all secret fields.
        Prevents accidental key exposure in logs, tracebacks, or debug output.
        """
        return (
            f"Settings("
            f"hf_api_token=***REDACTED***, "
            f"hf_embedding_model={self.hf_embedding_model!r}, "
            f"hf_llm_model={self.hf_llm_model!r}, "
            f"hf_reranker_model={self.hf_reranker_model!r}, "
            f"pinecone_api_key=***REDACTED***, "
            f"pinecone_index_name={self.pinecone_index_name!r}, "
            f"pinecone_host={self.pinecone_host!r}, "
            f"top_k={self.top_k}, "
            f"max_new_tokens={self.max_new_tokens}"
            f")"
        )


def get_settings() -> Settings:
    def _require(key: str) -> str:
        """Load a required env var; raise immediately with a clear message if absent."""
        val = os.getenv(key, "").strip()
        if not val:
            raise RuntimeError(
                f"Required environment variable '{key}' is missing or empty. "
                "Set it in your .env file (local) or Azure App Service Application Settings (production)."
            )
        return val

    return Settings(
        hf_api_token=_require("HF_API_TOKEN"),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hf_llm_model=os.getenv(
            "HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras"
        ),
        hf_reranker_model=os.getenv(
            "HF_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index_name=_require("PINECONE_INDEX_NAME"),
        pinecone_host=_require("PINECONE_HOST"),
        top_k=int(os.getenv("TOP_K", "8")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
    )
