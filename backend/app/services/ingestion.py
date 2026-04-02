"""
Ingestion pipeline:
  PDF → extract text → chunk → embed → store in ChromaDB

Why PyMuPDF (fitz) over other PDF parsers:
  Medical papers are often two-column layouts. PyMuPDF reads them
  in the correct reading order. PyPDF often scrambles columns.
"""
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import BinaryIO

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.config import settings
from app.core.vector_store import get_vector_store


def _make_doc_id(filename: str, content: bytes) -> str:
    """
    Stable unique ID for a document based on its content.
    Same file uploaded twice gets the same ID — prevents duplicates.
    """
    h = hashlib.sha256(content).hexdigest()[:16]
    safe_name = Path(filename).stem[:20].replace(" ", "_")
    return f"{safe_name}_{h}"


def _extract_text_from_pdf(content: bytes) -> list[dict]:
    """
    Extract text page by page using PyMuPDF.
    Uses 'blocks' mode to correctly handle two-column medical layouts.
    Each block is a paragraph-like unit — we join them with double
    newlines so the splitter sees proper paragraph boundaries.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        # Get text as blocks — each block is a text rectangle
        # sorted top-to-bottom, left-to-right within each column
        blocks = page.get_text("blocks", sort=True)

        # Each block is: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text, 1 = image — we skip images
        text_blocks = [
            b[4].strip()
            for b in blocks
            if b[6] == 0 and len(b[4].strip()) > 20
        ]

        # Join blocks with double newline — gives the splitter
        # clean paragraph boundaries to split on
        full_text = "\n\n".join(text_blocks)

        if len(full_text.strip()) > 50:
            pages.append({
                "page_number": page_num,
                "text": full_text,
            })

    doc.close()
    return pages


def _chunk_pages(pages: list[dict], doc_id: str, filename: str) -> list[Document]:
    """
    Split pages into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks
    first (\n\n), then line breaks (\n), then sentences ('. '),
    then words — in that order. This keeps medical sentences intact
    as much as possible rather than cutting mid-sentence.

    chunk_size=1000:  fits ~1 dense medical paragraph
    chunk_overlap=200: ~2 sentences of overlap so context isn't
                       lost at chunk boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(page_chunks):
            # Skip chunks that are just whitespace or very short
            if len(chunk_text.strip()) < 30:
                continue
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page["page_number"],
                    "chunk_index": i,
                }
            ))
    return chunks


def ingest_pdf(file: BinaryIO, filename: str) -> dict:
    """
    Full ingestion pipeline for one PDF.
    Returns metadata about what was ingested.
    """
    content = file.read()
    doc_id = _make_doc_id(filename, content)

    # Check if this exact document is already in the store
    vs = get_vector_store()
    existing = vs.get(where={"doc_id": doc_id})
    if existing["ids"]:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "already_exists",
            "chunk_count": len(existing["ids"]),
        }

    # Extract → chunk → store
    pages = _extract_text_from_pdf(content)
    if not pages:
        raise ValueError(f"No readable text found in {filename}. "
                         "The PDF might be scanned images without OCR.")

    chunks = _chunk_pages(pages, doc_id, filename)
    if not chunks:
        raise ValueError(f"Could not create chunks from {filename}.")

    # Give each chunk a stable unique ID
    chunk_ids = [f"{doc_id}_p{c.metadata['page']}_c{c.metadata['chunk_index']}"
                 for c in chunks]

    vs.add_documents(chunks, ids=chunk_ids)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "status": "ingested",
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def list_documents() -> list[dict]:
    """
    List all documents currently in the vector store.
    Aggregates chunk-level metadata back into document-level summaries.
    """
    vs = get_vector_store()
    result = vs.get(include=["metadatas"])

    # Aggregate chunk metadata back into a document-level summary.
    docs: dict[str, dict] = {}
    for meta in result["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "chunk_count": 0,
                "page_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1
        docs[doc_id]["page_count"] = max(
            docs[doc_id]["page_count"],
            meta.get("page", 0)
        )

    return list(docs.values())


def delete_document(doc_id: str) -> bool:
    """Remove all chunks belonging to a document."""
    vs = get_vector_store()
    existing = vs.get(where={"doc_id": doc_id})
    if not existing["ids"]:
        return False
    vs.delete(ids=existing["ids"])
    return True