"""API schemas for Azure RAG backend."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    page_count: int = 0
    chunk_count: int = 0
    status: str = "ingested"
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id: str


class SourceResult(BaseModel):
    source_id: str
    filename: str
    page: int
    doc_id: str
    score: float
    text: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = None
    doc_ids: Optional[list[str]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceResult]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    page_count: int = 0
    chunk_count: int = 0
    uploaded_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
