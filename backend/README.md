# Backend

FastAPI service for the Narayan medical research assistant.

## What it does

The backend accepts PDF uploads, extracts and chunks text, stores embeddings in ChromaDB, and answers questions with cited source excerpts through a retrieval-augmented generation pipeline.

## Runtime flow

1. Upload a PDF through the document API.
2. The ingestion service extracts text with PyMuPDF, chunks it, and stores the chunks in ChromaDB.
3. The query API retrieves the most relevant chunks, reranks them, and sends them to the LLM with a strict citation prompt.
4. The API returns the answer, source metadata, and token usage.

## Key modules

- [app/main.py](app/main.py) wires the FastAPI app, CORS, and startup warmup.
- [app/api/documents.py](app/api/documents.py) handles upload, list, and delete operations.
- [app/api/query.py](app/api/query.py) exposes the question-answering endpoint.
- [app/services/ingestion.py](app/services/ingestion.py) performs PDF extraction, chunking, and indexing.
- [app/services/rag.py](app/services/rag.py) performs retrieval, reranking, and answer generation.
- [app/core/config.py](app/core/config.py) holds environment-backed settings.
- [app/core/vector_store.py](app/core/vector_store.py) manages the shared embedding model and persistent Chroma collection.
- [app/models/schemas.py](app/models/schemas.py) defines request and response models.

## Environment variables

Set these in a local `.env` file or the environment before running the service.

- `OPENROUTER_API_KEY`: API key used by the chat model client.
- `LLM_MODEL`: Model name passed to the chat completion API.
- `EMBEDDING_MODEL`: Sentence-transformers model used for embeddings.
- `VECTOR_STORE_PATH`: On-disk location for the Chroma database.
- `COLLECTION_NAME`: Chroma collection name.
- `CHUNK_SIZE`: Target chunk length in characters.
- `CHUNK_OVERLAP`: Overlap between adjacent chunks.
- `TOP_K`: Number of chunks to retrieve before reranking.
- `RERANK_TOP_K`: Number of chunks sent to the LLM.
- `CORS_ORIGINS`: Allowed frontend origins.
- `MAX_FILE_SIZE_MB`: Maximum upload size accepted by the API.

## Running locally

Install dependencies and start the app from the backend directory.

```bash
uvicorn app.main:app --reload
```

The health check is available at `GET /health`.

## API surface

- `POST /api/documents/upload` uploads and indexes a PDF.
- `GET /api/documents/` lists ingested documents.
- `DELETE /api/documents/{doc_id}` removes a document and its chunks.
- `POST /api/query/` answers a question using the indexed sources.
