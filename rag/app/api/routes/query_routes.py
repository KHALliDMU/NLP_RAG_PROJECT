from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.controllers.query_controller import QueryController
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api", tags=["RAG"])


def get_rag_service() -> RAGService:
    """Get RAG service from app context."""
    from app.main import _rag_service
    if _rag_service is None:
        raise RuntimeError("RAG service not initialized")
    return _rag_service


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the CVs.")
    top_k: Optional[int] = Field(
        default=5, ge=1, le=20, description="Number of CV chunks to retrieve (1-20)."
    )
    score_threshold: Optional[float] = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum relevance score (0-1)."
    )


class SourceItem(BaseModel):
    file_name: str
    page: Optional[int]
    chunk_id: Optional[Any]
    chunk_index: Optional[int]
    total_chunks: Optional[int]
    upload_time: Optional[str]
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    num_chunks_used: int


@router.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Main RAG endpoint.

    - Embeds the question using the project embedding model.
    - Retrieves the top-k most relevant CV chunks from ChromaDB.
    - Injects chunks into a Groq LLM prompt and returns the generated answer.
    """
    try:
        controller = QueryController(rag_service)
        result = controller.query(
            question=body.question,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )

        sources = [SourceItem(**src) for src in result["sources"]]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            num_chunks_used=result["num_chunks_used"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
