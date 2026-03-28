"""RAG orchestration and prompt shaping."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.core.config import settings
from app.services import memory
from app.services.azure_openai import chat_completion, stream_chat_completion
from app.services.azure_search import hybrid_search
from app.services.cost_tracker import estimate_request_cost


SYSTEM_PROMPT = """You are Narayan, an Azure-native RAG assistant.
Answer only from provided context.
Use inline citations like [Source 1].
If context is insufficient, say so directly."""


def select_context_chunks(chunks: Sequence[Mapping[str, Any]], max_chars: int = 12_000) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "id": chunk.get("id", ""),
                "doc_id": chunk.get("doc_id", ""),
                "filename": chunk.get("filename", ""),
                "page": int(chunk.get("page", 0) or 0),
                "chunk_index": int(chunk.get("chunk_index", 0) or 0),
                "content": str(chunk.get("content", "")).strip(),
                "score": float(chunk.get("score", 0.0) or 0.0),
            }
            for chunk in chunks
            if str(chunk.get("content", "")).strip()
        ),
        key=lambda item: (item["score"], -item["chunk_index"]),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    used = 0
    for chunk in ranked:
        content = chunk["content"]
        chunk_len = len(content)
        if selected and used + chunk_len > max_chars:
            continue
        if not selected and chunk_len > max_chars:
            chunk = dict(chunk)
            chunk["content"] = content[:max_chars]
            selected.append(chunk)
            break
        selected.append(chunk)
        used += chunk_len
        if used >= max_chars:
            break
    return selected


def _format_sources(chunks: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        label = f"[Source {index}]"
        header = f"{label} {chunk.get('filename', 'unknown')} page {chunk.get('page', '?')}"
        parts.append(f"{header}\n{chunk.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def build_rag_messages(
    question: str,
    context_chunks: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, str]],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    selected = select_context_chunks(context_chunks, max_chars=settings.MAX_CONTEXT_CHARS)
    history_window = list(history)[-settings.MAX_HISTORY_MESSAGES :]
    compacted_history = memory.summarize_history(
        [dict(item) for item in history_window],
        keep_last=settings.MEMORY_SUMMARY_WINDOW,
    )

    context_block = _format_sources(selected)
    prompt = (
        "Context sources:\n"
        f"{context_block}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer with direct citations like [Source 1]."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(compacted_history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _format_source_results(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("content", ""))
        results.append(
            {
                "source_id": f"Source {index}",
                "doc_id": chunk.get("doc_id", ""),
                "filename": chunk.get("filename", ""),
                "page": int(chunk.get("page", 0) or 0),
                "score": float(chunk.get("score", 0.0) or 0.0),
                "text": text if len(text) <= 400 else f"{text[:400]}...",
            }
        )
    return results


def rerank_chunks(chunks: Sequence[Mapping[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Second retrieval stage.

    Stage one over-fetches a wide candidate pool from Azure AI Search; this
    stage reranks those candidates by their hybrid (vector + keyword) relevance
    score and keeps only the top ``top_k`` chunks the LLM actually sees. Slicing
    a wider pool down to a few chunks gives the model tighter, higher-signal
    context and keeps citations focused.
    """
    ranked = sorted(
        chunks,
        key=lambda chunk: float(chunk.get("score", 0.0) or 0.0),
        reverse=True,
    )
    return [dict(chunk) for chunk in ranked[:top_k]]


def run_rag(question: str, session_id: str, doc_ids: Sequence[str] | None = None, top_k: int | None = None) -> dict[str, Any]:
    candidates = hybrid_search(question, top_k=settings.RETRIEVAL_CANDIDATES, doc_ids=doc_ids)
    reranked = rerank_chunks(candidates, top_k or settings.RERANK_TOP_K)
    selected = select_context_chunks(reranked, max_chars=settings.MAX_CONTEXT_CHARS)
    history = memory.get_history(session_id)
    messages = build_rag_messages(question, selected, history)

    response = chat_completion(messages, max_tokens=settings.ANSWER_MAX_TOKENS)
    answer = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    estimated_cost_usd = estimate_request_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=settings.AZURE_OPENAI_CHAT_MODEL_NAME,
    )

    memory.save_message(session_id, "user", question)
    memory.save_message(session_id, "assistant", answer)

    return {
        "answer": answer,
        "sources": _format_source_results(selected),
        "messages": messages,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "model": settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        "session_id": session_id,
    }


def stream_rag(question: str, session_id: str, doc_ids: Sequence[str] | None = None, top_k: int | None = None):
    candidates = hybrid_search(question, top_k=settings.RETRIEVAL_CANDIDATES, doc_ids=doc_ids)
    reranked = rerank_chunks(candidates, top_k or settings.RERANK_TOP_K)
    selected = select_context_chunks(reranked, max_chars=settings.MAX_CONTEXT_CHARS)
    history = memory.get_history(session_id)
    messages = build_rag_messages(question, selected, history)

    answer_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for event in stream_chat_completion(messages, max_tokens=settings.ANSWER_MAX_TOKENS):
        if event["type"] == "delta":
            answer_parts.append(event["text"])
            yield {"type": "delta", "text": event["text"]}
        elif event["type"] == "done":
            usage = event["usage"]
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)

    answer = "".join(answer_parts)
    memory.save_message(session_id, "user", question)
    memory.save_message(session_id, "assistant", answer)
    yield {
        "type": "done",
        "answer": answer,
        "sources": _format_source_results(selected),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimate_request_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=settings.AZURE_OPENAI_CHAT_MODEL_NAME,
        ),
        "model": settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        "session_id": session_id,
    }
