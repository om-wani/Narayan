"""FastAPI app entrypoint for Azure RAG backend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes import chat, ingest
from app.utils.logger import configure_logging, get_logger


logger = get_logger("narayan.azure.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Azure RAG backend starting")
    yield
    logger.info("Azure RAG backend stopping")


app = FastAPI(
    title=settings.APP_NAME,
    description="Azure-native RAG system with Azure OpenAI, Azure AI Search, SSE, memory, and safety.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}
