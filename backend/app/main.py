from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import documents, query
from app.core.config import settings
from app.core.vector_store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and embedding model at startup."""
    print("Starting up — loading models...")
    get_vector_store()
    print("Ready.")
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
    return {"status": "ok"}