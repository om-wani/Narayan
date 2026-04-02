"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Typed configuration for model, retrieval, and API settings."""

    # LLM configuration
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "google/gemma-3-27b-it:free"

    # Local embedding model used for document chunks and queries
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Persistent ChromaDB storage on disk
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "medical_papers"

    # Chunking tuned for dense medical-paper prose
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval defaults for the RAG pipeline
    TOP_K: int = 5
    RERANK_TOP_K: int = 3

    # API/server configuration
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    MAX_FILE_SIZE_MB: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()