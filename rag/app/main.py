"""Main FastAPI application."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import query_routes, upload_routes
from app.services.rag_service import RAGService

# Global RAG service instance
_rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown."""
    global _rag_service
    _rag_service = RAGService()
    try:
        _rag_service.initialize()
        print("RAG service initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize RAG service: {e}")
    yield
    _rag_service = None


app = FastAPI(
    title="CV RAG API",
    description="Retrieval-Augmented Generation over a ChromaDB CV corpus.",
    version="1.1.0",
    lifespan=lifespan,
)

# Register route modules
# app.include_router(health_routes.router)
app.include_router(query_routes.router)
app.include_router(upload_routes.router)


@app.get("/", tags=["Info"])
def root():
    """API documentation and endpoints."""
    return {
        "message": "CV RAG API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "query": "POST /api/query",
            "upload": "POST /api/upload",
        },
    }
