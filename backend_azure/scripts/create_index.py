"""Create Azure AI Search index for Narayan Azure RAG."""

from __future__ import annotations

import json

from app.core.config import settings


def build_index_definition() -> dict[str, object]:
    return {
        "name": settings.AZURE_SEARCH_INDEX_NAME,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True, "sortable": True},
            {"name": "doc_id", "type": "Edm.String", "filterable": True, "facetable": True},
            {"name": "filename", "type": "Edm.String", "searchable": True, "filterable": True},
            {"name": "page", "type": "Edm.Int32", "filterable": True, "sortable": True},
            {"name": "chunk_index", "type": "Edm.Int32", "filterable": True, "sortable": True},
            {"name": "content", "type": "Edm.String", "searchable": True},
            {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": settings.EMBEDDING_DIMENSIONS,
                "vectorSearchProfile": "narayan-vector-profile",
            },
            {"name": "uploaded_at", "type": "Edm.DateTimeOffset", "filterable": True, "sortable": True},
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "narayan-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {"metric": "cosine"},
                }
            ],
            "profiles": [
                {
                    "name": "narayan-vector-profile",
                    "algorithm": "narayan-hnsw",
                }
            ],
        },
    }


def main() -> None:
    import httpx

    endpoint = settings.AZURE_SEARCH_ENDPOINT.rstrip("/")
    api_key = settings.AZURE_SEARCH_API_KEY
    url = f"{endpoint}/indexes/{settings.AZURE_SEARCH_INDEX_NAME}?api-version={settings.AZURE_SEARCH_API_VERSION}"
    payload = build_index_definition()

    with httpx.Client(timeout=60.0, headers={"Content-Type": "application/json", "api-key": api_key}) as client:
        response = client.put(url, content=json.dumps(payload))
        response.raise_for_status()
        print(f"Created or updated index: {settings.AZURE_SEARCH_INDEX_NAME}")


if __name__ == "__main__":
    main()
