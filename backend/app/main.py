"""FastAPI application entrypoint for the backend.

The app wires the document-ingestion and question-answering routers,
configures CORS for the frontend, and preloads the vector store during
startup so the first request does not pay model-loading latency.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import documents, query
from app.core.config import settings
from app.core.vector_store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the embedding model and persistent vector store on startup."""
    print("Starting up: loading embeddings and vector store...")
    get_vector_store()
    print("Startup complete.")
    yield


app = FastAPI(
    title="Medical Research RAG",
    description="Upload medical papers, ask questions, get cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["documents"],
)
app.include_router(
    query.router,
    prefix="/api/query",
    tags=["query"],
)


@app.get("/health")
async def health():
    """Lightweight liveness check used by deploy and monitoring probes."""
    return {"status": "ok"}