"""Document ingestion routes."""

from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.schemas import DeleteResponse, DocumentListResponse, DocumentMetadata, IngestResponse
from app.services.ingestion import delete_document, ingest_pdf, list_documents


router = APIRouter()
ALLOWED_TYPES = {"application/pdf", "binary/octet-stream", "application/octet-stream"}


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    if file.content_type not in ALLOWED_TYPES and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Got: {file.content_type}")

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max size is {settings.MAX_UPLOAD_MB}MB.")

    try:
        result = ingest_pdf(io.BytesIO(content), file.filename or "upload.pdf")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(**result)


@router.get("/", response_model=DocumentListResponse)
async def get_documents() -> DocumentListResponse:
    documents = [
        DocumentMetadata(**document)
        for document in list_documents()
    ]
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str) -> DeleteResponse:
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DeleteResponse(deleted=True, doc_id=doc_id)
