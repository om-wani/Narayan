"""
RAG query pipeline:
  question → embed → retrieve → rerank → generate → cited answer

The flow in plain english:
  1. Embed the question into a vector
  2. Find the top-K most similar chunks in ChromaDB
  3. Rerank them — keep only the best 3 for the LLM
  4. Format those 3 chunks as numbered sources
  5. Send question + sources to the LLM
  6. Return the answer with citations
"""
from openai import OpenAI

from app.core.config import settings
from app.core.vector_store import get_vector_store


# ── System prompt ─────────────────────────────────────────────────
# This is what shapes the LLM's behaviour for medical research use.
# "Only use the provided sources" is the key hallucination guard.

SYSTEM_PROMPT = """You are a precise medical research assistant.

You will be given numbered source excerpts from research papers and a question.

STRICT RULES:
- If sources contain relevant information: answer directly using [Source N] citations.
- ONLY say "The provided documents don't contain enough information" if the sources
  are completely unrelated to the question. Never say it as a preamble.
- Start your answer immediately. No preamble, no meta-commentary.
- Use correct medical terminology. Be precise and concise."""


def _format_sources(chunks: list) -> str:
    """
    Format retrieved chunks into a numbered source list for the prompt.
    Each source shows its filename and page so the LLM can cite correctly.
    """
    parts = []
    for i, (doc, score) in enumerate(chunks, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Source {i}] {filename}, page {page} "
            f"(relevance: {score:.2f})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _rerank(chunks_with_scores: list, top_k: int) -> list:
    """
    Simple rerank by score — keeps the highest-scoring chunks.

    This is 'naive reranking' — just sorting by the vector similarity
    score ChromaDB already computed. It works well enough for a v1.

    Phase 2 upgrade: replace this with a cross-encoder model or the
    Cohere rerank API for significantly better results. A cross-encoder
    reads the question AND each chunk together, rather than comparing
    them separately, which catches relevance the vector search misses.
    """
    sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_chunks[:top_k]


def answer_question(
    question: str,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Full RAG pipeline. Returns a dict with answer, sources, and metadata.

    Args:
        question: The user's question
        doc_ids:  Optional list of doc_ids to search within.
                  None means search across all documents.
        top_k:    How many chunks to retrieve. Defaults to settings.TOP_K.
    """
    vs = get_vector_store()
    fetch_k = top_k or settings.TOP_K

    # ── 1. Retrieve ────────────────────────────────────────────────
    # Build an optional filter to scope search to specific documents
    where_filter = None
    if doc_ids and len(doc_ids) == 1:
        where_filter = {"doc_id": doc_ids[0]}
    elif doc_ids and len(doc_ids) > 1:
        where_filter = {"$or": [{"doc_id": d} for d in doc_ids]}

    # similarity_search_with_relevance_scores returns (Document, score) pairs
    # Score is cosine similarity: 1.0 = identical, 0.0 = completely different
    results = vs.similarity_search_with_relevance_scores(
        question,
        k=fetch_k * 2,        # over-fetch so reranking has candidates to work with
        filter=where_filter,
    )

    if not results:
        return {
            "answer": "The provided documents don't contain enough information "
                      "to answer this question.",
            "sources": [],
            "model": settings.LLM_MODEL,
        }

    # ── 2. Rerank ──────────────────────────────────────────────────
    top_chunks = _rerank(results, settings.RERANK_TOP_K)

    # ── 3. Format context ──────────────────────────────────────────
    sources_text = _format_sources(top_chunks)
    user_prompt = (
        f"Source excerpts from research papers:\n\n"
        f"{sources_text}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite sources inline using [Source N] notation):"
    )

    # ── 4. Generate ────────────────────────────────────────────────
    # OpenRouter uses the OpenAI SDK format with a custom base_url
    client = OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="http://localhost:11434/v1",
    )

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0

    # ── 5. Build response ──────────────────────────────────────────
    # Map chunks to clean source dicts for the API response
    source_list = [
        {
            "filename": doc.metadata.get("filename", "unknown"),
            "page":     doc.metadata.get("page", "?"),
            "doc_id":   doc.metadata.get("doc_id", ""),
            "score":    round(float(score), 3),
            "text":     doc.page_content[:300] + "..."
                        if len(doc.page_content) > 300
                        else doc.page_content,
        }
        for doc, score in top_chunks
    ]

    return {
        "answer":      answer_text,
        "sources":     source_list,
        "model":       settings.LLM_MODEL,
        "tokens_used": tokens_used,
    }