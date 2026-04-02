"""Question-answering API for the RAG pipeline."""

from fastapi import APIRouter, HTTPException
from app.services.rag import answer_question
from app.models.schemas import QueryRequest, QueryResponse, SourceResult

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run retrieval-augmented generation for a user question."""
    try:
        result = answer_question(
            question=req.question,
            doc_ids=req.doc_ids,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceResult(**s) for s in result["sources"]],
        model=result["model"],
        tokens_used=result.get("tokens_used", 0),
    )