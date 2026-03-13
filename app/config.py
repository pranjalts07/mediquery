"""
app/config.py
Centralised configuration loaded from environment variables.
All secrets are injected at runtime — never hardcoded.
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

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_host: str

    # App
    top_k: int
    max_new_tokens: int


def get_settings() -> Settings:
    def _require(key: str) -> str:
        val = os.getenv(key, "").strip()
        if not val:
            raise RuntimeError(
                f"Required environment variable '{key}' is missing or empty. "
                "Set it in your .env file or Azure App Service Application Settings."
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
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index_name=_require("PINECONE_INDEX_NAME"),
        pinecone_host=_require("PINECONE_HOST"),
        top_k=int(os.getenv("TOP_K", "4")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
    )