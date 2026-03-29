# Narayan Azure Backend

Azure-native RAG version of Narayan. Keep current `backend/` folder intact. Use this folder for Azure OpenAI + Azure AI Search work.

## What this version does

- Upload PDF docs
- Extract and chunk text
- Embed chunks with Azure OpenAI
- Index chunks in Azure AI Search
- Answer with two-stage retrieval (over-fetch 10 candidates, rerank to top 3) and inline citations
- Stream chat responses through SSE
- Track token usage and estimated cost
- Keep short conversation memory
- Apply simple prompt-injection safety checks

## Azure setup

You need one Azure student subscription. No separate OpenAI or Search account.

Create:

- Azure OpenAI resource in Azure AI Foundry
- Chat deployment, usually cheap model like `gpt-4o-mini`
- Embedding deployment, usually `text-embedding-3-small`
- Azure AI Search resource

Budget:

- Set cost alert in Azure portal
- Keep this project on student credit only
- Start with free or lowest tier you can use for Search

## Local setup

1. Copy env file.

```bash
cp .env.example .env
```

2. Fill Azure values in `.env`.

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create search index.

```bash
python scripts/create_index.py
```

5. Run app.

```bash
uvicorn app.main:app --reload
```

## Environment variables

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_CHAT_MODEL_NAME`
- `AZURE_OPENAI_EMBEDDING_MODEL_NAME`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`
- `AZURE_SEARCH_API_VERSION`
- `CORS_ORIGINS`
- `MAX_UPLOAD_MB`
- `MAX_CONTEXT_CHARS`
- `MAX_HISTORY_MESSAGES`
- `MEMORY_SUMMARY_WINDOW`
- `TOP_K`
- `RETRIEVAL_CANDIDATES`
- `RERANK_TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `ANSWER_MAX_TOKENS`
- `BUDGET_ALERT_USD`

## API

- `POST /api/ingest/upload`
- `GET /api/ingest/`
- `DELETE /api/ingest/{doc_id}`
- `POST /api/chat/`
- `POST /api/chat/stream`
- `GET /health`
