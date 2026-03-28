"""Chat routes with JSON and SSE response modes."""

from __future__ import annotations

import json
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse, SourceResult
from app.services.rag_pipeline import run_rag, stream_rag
from app.services.safety import assert_safe_text


router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    try:
        assert_safe_text(req.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start = time.perf_counter()
    try:
        result = run_rag(req.question, session_id=session_id, doc_ids=req.doc_ids, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.perf_counter() - start) * 1000)

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        sources=[SourceResult(**source) for source in result["sources"]],
        model=result["model"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
        estimated_cost_usd=result["estimated_cost_usd"],
        latency_ms=latency_ms,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    session_id = req.session_id or str(uuid4())
    try:
        assert_safe_text(req.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        try:
            for event in stream_rag(req.question, session_id=session_id, doc_ids=req.doc_ids, top_k=req.top_k):
                yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
