from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # OpenRouter — LLM calls
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "google/gemma-3-27b-it:free"

    # Embeddings — runs locally, no API needed
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector store — just a folder on disk
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "medical_papers"

    # Chunking — tuned for medical papers
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K: int = 5
    RERANK_TOP_K: int = 3

    # API
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    MAX_FILE_SIZE_MB: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()