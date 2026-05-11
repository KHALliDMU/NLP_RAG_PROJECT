"""Query controller — handles RAG query business logic."""
from __future__ import annotations

from app.services.rag_service import RAGService


class QueryController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> dict:
        """Execute RAG query with optional filtering."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not self.rag_service.is_ready():
            raise RuntimeError("Vector DB not initialized or empty")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")

        if score_threshold < 0.0 or score_threshold > 1.0:
            raise ValueError("score_threshold must be between 0 and 1")

        return self.rag_service.query(
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
        )
