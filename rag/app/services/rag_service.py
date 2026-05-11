"""RAG service — handles core RAG operations."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = "vector_db"
CHROMA_COLLECTION = "cv_collection"
LLM_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are an expert HR assistant. Use ONLY the following CV excerpts to answer the question.
If the information is not in the provided context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGService:
    def __init__(self):
        self.vector_db = None
        self.embeddings = None

    def initialize(self):
        """Initialize vector DB and embeddings."""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db = self._load_vector_db()

    def _load_vector_db(self) -> Chroma:
        """Load the persisted ChromaDB store."""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
        )

    def build_rag_chain(self, top_k: int = 5):
        """Build a LangChain retrieval chain."""
        if self.vector_db is None:
            raise RuntimeError("Vector DB not initialized")

        retriever = self.vector_db.as_retriever(search_kwargs={"k": top_k})
        llm = ChatGroq(
            model=LLM_MODEL,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2,
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT_TEMPLATE
            | llm
            | StrOutputParser()
        )
        return chain, retriever

    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.0) -> dict[str, Any]:
        """Execute RAG query with optional score threshold."""
        if self.vector_db is None:
            raise RuntimeError("Vector DB not initialized")

        chain, _ = self.build_rag_chain(top_k=top_k)
        answer = chain.invoke(question)

        docs_with_scores = self.vector_db.similarity_search_with_relevance_scores(
            question, k=top_k
        )

        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in docs_with_scores if score >= score_threshold
        ]

        sources = []
        for doc, score in filtered_results:
            meta = doc.metadata
            sources.append(
                {
                    "file_name": meta.get("file_name", meta.get("source", "unknown")),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "total_chunks": meta.get("total_chunks"),
                    "upload_time": meta.get("upload_time"),
                    "score": round(float(score), 4),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_used": len(sources),
        }

    def add_documents(self, chunks: list, file_name: str, upload_time: str | None = None):
        """Add new chunks to the vector DB with enhanced metadata."""
        if self.vector_db is None:
            raise RuntimeError("Vector DB not initialized")

        upload_time = upload_time or datetime.now().isoformat()
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_chunks
            chunk.metadata["upload_time"] = upload_time
            chunk.metadata["file_name"] = file_name

        self.vector_db.add_documents(chunks)
        return {
            "file_name": file_name,
            "chunks_added": len(chunks),
            "upload_time": upload_time,
        }

    def is_ready(self) -> bool:
        """Check if vector DB is ready."""
        if self.vector_db is None:
            return False
        try:
            return self.vector_db.get()["ids"] is not None
        except Exception:
            return False
