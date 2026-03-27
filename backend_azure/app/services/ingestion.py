"""PDF ingestion into Azure AI Search."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.azure_openai import embed_texts
from app.services.azure_search import delete_documents, find_document_chunk_ids, get_all_documents, search_documents, upsert_documents


def _make_doc_id(filename: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()[:16]
    safe_name = Path(filename).stem[:20].replace(" ", "_")
    return f"{safe_name}_{digest}"


def _extract_pages(content: bytes) -> list[dict[str, object]]:
    document = fitz.open(stream=content, filetype="pdf")
    pages: list[dict[str, object]] = []
    for page_number, page in enumerate(document, start=1):
        blocks = page.get_text("blocks", sort=True)
        text_blocks = [block[4].strip() for block in blocks if block[6] == 0 and len(block[4].strip()) > 20]
        page_text = "\n\n".join(text_blocks).strip()
        if len(page_text) > 50:
            pages.append({"page_number": page_number, "text": page_text})
    document.close()
    return pages


def _chunk_pages(pages: list[dict[str, object]], doc_id: str, filename: str) -> list[dict[str, object]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, object]] = []
    for page in pages:
        page_text = str(page["text"])
        page_number = int(page["page_number"])
        for chunk_index, chunk_text in enumerate(splitter.split_text(page_text)):
            content = chunk_text.strip()
            if len(content) < 30:
                continue
            chunks.append(
                {
                    "id": f"{doc_id}_p{page_number}_c{chunk_index}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "content": content,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return chunks


def ingest_pdf(file: BinaryIO, filename: str) -> dict[str, object]:
    content = file.read()
    doc_id = _make_doc_id(filename, content)

    existing = search_documents("*", top_k=1000, doc_ids=[doc_id], use_vector=False)
    if existing:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "already_exists",
            "chunk_count": len(existing),
        }

    pages = _extract_pages(content)
    if not pages:
        raise ValueError("No readable text found in PDF.")

    chunks = _chunk_pages(pages, doc_id, filename)
    if not chunks:
        raise ValueError("Could not build chunks from PDF.")

    embeddings = embed_texts([str(chunk["content"]) for chunk in chunks])
    payload: list[dict[str, object]] = []
    for chunk, vector in zip(chunks, embeddings, strict=True):
        payload.append(
            {
                **chunk,
                "content_vector": vector,
            }
        )

    upsert_documents(payload)
    return {
        "doc_id": doc_id,
        "filename": filename,
        "status": "ingested",
        "page_count": len(pages),
        "chunk_count": len(payload),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def list_documents() -> list[dict[str, object]]:
    return get_all_documents()


def delete_document(doc_id: str) -> bool:
    chunk_ids = find_document_chunk_ids(doc_id)
    if not chunk_ids:
        return False
    delete_documents(chunk_ids)
    return True
