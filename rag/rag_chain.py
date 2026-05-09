"""
rag_chain.py — Core RAG logic for CV retrieval and answer generation.

Usage (standalone test):
    python rag_chain.py
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

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

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_vector_db() -> Chroma:
    """Load the persisted ChromaDB store with the project embedding model."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def build_rag_chain(vector_db: Chroma, top_k: int = 5):
    """
    Build a LangChain retrieval chain.

    Steps:
        1. Retriever  — ChromaDB similarity search (top-k)
        2. Prompt     — Injects retrieved chunks + user question
        3. LLM        — Google Gemini generates the answer
        4. Parser     — Returns a plain string
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
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


def query_rag(question: str, top_k: int = 5) -> dict[str, Any]:
    """
    End-to-end RAG query.

    Returns:
        {
            "answer":          str,
            "sources":         [{"file_name", "page", "chunk_id", "score"}, ...],
            "num_chunks_used": int,
        }
    """
    vector_db = load_vector_db()
    chain, retriever = build_rag_chain(vector_db, top_k=top_k)

    # Run the chain for the answer
    answer = chain.invoke(question)

    # Retrieve docs separately to extract metadata + scores
    docs_with_scores = vector_db.similarity_search_with_relevance_scores(
        question, k=top_k
    )

    sources = []
    for doc, score in docs_with_scores:
        meta = doc.metadata
        sources.append(
            {
                "file_name": meta.get("file_name", meta.get("source", "unknown")),
                "page": meta.get("page", None),
                "chunk_id": meta.get("chunk_id", None),
                "score": round(float(score), 4),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "num_chunks_used": len(sources),
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TEST_QUERY = "Find candidates with Python experience"
    print(f"Query: {TEST_QUERY}\n{'-' * 60}")

    result = query_rag(TEST_QUERY, top_k=5)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Chunks used: {result['num_chunks_used']}")
    print("Sources:")
    for src in result["sources"]:
        print(f"  • {src['file_name']}  page={src['page']}  score={src['score']}")
