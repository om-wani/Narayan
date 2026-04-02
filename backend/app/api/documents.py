"""Document management API.

This router handles PDF upload, document listing, and deletion. It is a thin
HTTP layer over the ingestion service so the validation rules stay close to the
request boundary while the indexing logic remains testable in isolation.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.ingestion import ingest_pdf, list_documents, delete_document
from app.models.schemas import DocumentMetadata, DocumentListResponse, DeleteResponse
from app.core.config import settings
import io

router = APIRouter()

ALLOWED_TYPES = {"application/pdf", "binary/octet-stream"}
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentMetadata,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(file: UploadFile = File(...)):
    """Validate an uploaded PDF and ingest it into the vector store."""
    # Some browsers report PDFs as octet-stream, so accept both that and an
    # explicit .pdf extension to avoid rejecting legitimate uploads.
    if file.content_type not in ALLOWED_TYPES and not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Got: {file.content_type}"
        )

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {settings.MAX_FILE_SIZE_MB}MB."
        )

    try:
        result = ingest_pdf(io.BytesIO(content), file.filename or "upload.pdf")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return DocumentMetadata(
        doc_id=result["doc_id"],
        filename=result["filename"],
        page_count=result.get("page_count", 0),
        chunk_count=result["chunk_count"],
        uploaded_at=result.get("uploaded_at"),
    )


@router.get("/", response_model=DocumentListResponse)
async def get_documents():
    """Return document-level summaries derived from stored chunks."""
    docs = list_documents()
    return DocumentListResponse(
        documents=[
            DocumentMetadata(
                doc_id=d["doc_id"],
                filename=d["filename"],
                page_count=d.get("page_count", 0),
                chunk_count=d["chunk_count"],
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str):
    """Delete a document and every chunk associated with it."""
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found."
        )
    return DeleteResponse(deleted=True, doc_id=doc_id)