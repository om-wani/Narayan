"""Application settings for Azure RAG backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Narayan Azure RAG")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    AZURE_OPENAI_CHAT_MODEL_NAME: str = os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4o-mini")
    AZURE_OPENAI_EMBEDDING_MODEL_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")

    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "narayan-docs")
    AZURE_SEARCH_API_VERSION: str = os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01")

    CORS_ORIGINS_RAW: str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
    MEMORY_SUMMARY_WINDOW: int = int(os.getenv("MEMORY_SUMMARY_WINDOW", "4"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    # Two-stage retrieval: over-fetch RETRIEVAL_CANDIDATES chunks from Azure AI
    # Search, then rerank down to RERANK_TOP_K chunks for the LLM context.
    RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "10"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))
    ANSWER_MAX_TOKENS: int = int(os.getenv("ANSWER_MAX_TOKENS", "900"))
    BUDGET_ALERT_USD: float = float(os.getenv("BUDGET_ALERT_USD", "5.0"))

    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    # Chunking tuned for dense two-column academic papers: ~1 paragraph per
    # chunk with ~2 sentences of overlap so context survives chunk boundaries.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.CORS_ORIGINS_RAW.split(",") if item.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
