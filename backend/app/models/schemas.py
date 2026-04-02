"""Pydantic request and response models for the backend API."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ── Document models ───────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Document-level metadata returned after upload or during listing."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Container for document summaries and a total count."""

    documents: List[DocumentMetadata]
    total: int


class DeleteResponse(BaseModel):
    """Confirmation payload returned after deleting a document."""

    deleted: bool
    doc_id: str


# ── Query models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Input payload for a RAG question request."""

    question: str = Field(..., min_length=3, max_length=2000)
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Limit search to specific doc IDs. None = search all."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceResult(BaseModel):
    """Single retrieved source excerpt included with an answer."""

    filename: str
    page: int
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """Answer text plus the sources and metadata used to produce it."""

    answer: str
    sources: List[SourceResult]
    model: str
    tokens_used: int