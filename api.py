"""
api.py — FastAPI application exposing the RAG pipeline.

Start the server:
    uvicorn api:app --reload

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_chain import load_vector_db, build_rag_chain

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CV RAG API",
    description="Retrieval-Augmented Generation over a ChromaDB CV corpus.",
    version="1.0.0",
)

# Load the vector DB once at startup so every request shares the same instance.
_vector_db = None


@app.on_event("startup")
def startup_event():
    global _vector_db
    _vector_db = load_vector_db()


# ── Schemas ───────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the CVs.")
    top_k: Optional[int] = Field(
        default=5, ge=1, le=20, description="Number of CV chunks to retrieve (1-20)."
    )


class SourceItem(BaseModel):
    file_name: str
    page: Optional[int]
    chunk_id: Optional[Any]
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    num_chunks_used: int
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Utility"])
def health_check():
    """Returns service status and whether the vector DB is loaded."""
    return {
        "status": "ok",
        "vector_db_loaded": _vector_db is not None,
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_endpoint(body: QueryRequest):
    """
    Main RAG endpoint.

    - Embeds the question using the project embedding model.
    - Retrieves the top-k most relevant CV chunks from ChromaDB.
    - Injects chunks into a Groq LLM prompt and returns the generated answer.
    """
    if _vector_db is None:
        raise HTTPException(status_code=503, detail="Vector DB not yet loaded.")

    t0 = time.perf_counter()

    try:
        chain, _ = build_rag_chain(_vector_db, top_k=body.top_k)
        answer = chain.invoke(body.question)

        docs_with_scores = _vector_db.similarity_search_with_relevance_scores(
            body.question, k=body.top_k
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    sources = []
    for doc, score in docs_with_scores:
        meta = doc.metadata
        sources.append(
            SourceItem(
                file_name=meta.get("file_name", meta.get("source", "unknown")),
                page=meta.get("page"),
                chunk_id=meta.get("chunk_id"),
                score=round(float(score), 4),
            )
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        num_chunks_used=len(sources),
        latency_ms=latency_ms,
    )
