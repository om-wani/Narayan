"""Azure AI Search REST helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Sequence

from app.core.config import settings
from app.services.azure_openai import embed_text


def _require_config() -> None:
    required = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    missing = [name for name in required if not getattr(settings, name)]
    if missing:
        raise RuntimeError(f"Missing Azure Search config: {', '.join(missing)}")


@lru_cache(maxsize=1)
def _client() -> Any:
    _require_config()
    import httpx

    return httpx.Client(
        timeout=60.0,
        headers={
            "Content-Type": "application/json",
            "api-key": settings.AZURE_SEARCH_API_KEY,
        },
    )


def _base_url() -> str:
    return settings.AZURE_SEARCH_ENDPOINT.rstrip("/")


def _index_docs_url() -> str:
    return f"{_base_url()}/indexes/{settings.AZURE_SEARCH_INDEX_NAME}/docs/index?api-version={settings.AZURE_SEARCH_API_VERSION}"


def _search_url() -> str:
    return f"{_base_url()}/indexes/{settings.AZURE_SEARCH_INDEX_NAME}/docs/search?api-version={settings.AZURE_SEARCH_API_VERSION}"


def _escape_odata(value: str) -> str:
    return value.replace("'", "''")


def build_doc_filter(doc_ids: Sequence[str] | None) -> str | None:
    if not doc_ids:
        return None
    parts = [f"doc_id eq '{_escape_odata(doc_id)}'" for doc_id in doc_ids]
    return " or ".join(parts)


def upsert_documents(documents: Sequence[dict[str, Any]]) -> None:
    if not documents:
        return

    payload = {
        "value": [
            {"@search.action": "mergeOrUpload", **document}
            for document in documents
        ]
    }
    response = _client().post(_index_docs_url(), content=json.dumps(payload))
    response.raise_for_status()


def delete_documents(ids: Sequence[str]) -> None:
    if not ids:
        return

    payload = {
        "value": [{"@search.action": "delete", "id": doc_id} for doc_id in ids]
    }
    response = _client().post(_index_docs_url(), content=json.dumps(payload))
    response.raise_for_status()


def search_documents(
    query: str,
    *,
    top_k: int = 5,
    doc_ids: Sequence[str] | None = None,
    use_vector: bool = True,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "search": query,
        "top": top_k,
        "select": "id,doc_id,filename,page,chunk_index,content,uploaded_at",
    }

    filter_expression = build_doc_filter(doc_ids)
    if filter_expression:
        payload["filter"] = filter_expression

    if use_vector and query.strip() and query != "*":
        payload["vectorQueries"] = [
            {
                "kind": "vector",
                "vector": embed_text(query),
                "fields": "content_vector",
                "k": top_k,
            }
        ]

    response = _client().post(_search_url(), content=json.dumps(payload))
    response.raise_for_status()
    data = response.json()

    results: list[dict[str, Any]] = []
    for item in data.get("value", []):
        results.append(
            {
                "id": item.get("id", ""),
                "doc_id": item.get("doc_id", ""),
                "filename": item.get("filename", ""),
                "page": int(item.get("page", 0) or 0),
                "chunk_index": int(item.get("chunk_index", 0) or 0),
                "content": item.get("content", ""),
                "uploaded_at": item.get("uploaded_at"),
                "score": float(item.get("@search.score", 0.0) or 0.0),
            }
        )
    return results


def hybrid_search(query: str, *, top_k: int = 5, doc_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    return search_documents(query, top_k=top_k, doc_ids=doc_ids, use_vector=True)


def get_all_documents() -> list[dict[str, Any]]:
    items = search_documents("*", top_k=1000, use_vector=False)
    documents: dict[str, dict[str, Any]] = {}
    for item in items:
        doc_id = item["doc_id"] or item["id"]
        if doc_id not in documents:
            documents[doc_id] = {
                "doc_id": doc_id,
                "filename": item.get("filename", ""),
                "page_count": 0,
                "chunk_count": 0,
                "uploaded_at": item.get("uploaded_at"),
            }
        documents[doc_id]["chunk_count"] += 1
        documents[doc_id]["page_count"] = max(documents[doc_id]["page_count"], item.get("page", 0))
    return list(documents.values())


def find_document_chunk_ids(doc_id: str) -> list[str]:
    results = search_documents("*", top_k=1000, doc_ids=[doc_id], use_vector=False)
    return [item["id"] for item in results if item.get("id")]
