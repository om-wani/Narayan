from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ── Document models ───────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id: str


# ── Query models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Limit search to specific doc IDs. None = search all."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceResult(BaseModel):
    filename: str
    page: int
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResult]
    model: str
    tokens_used: int