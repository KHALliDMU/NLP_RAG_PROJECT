# RAG Project — Missing Features Plan

## Already Implemented

The project already has:

- FastAPI backend
- Basic RAG pipeline
- PDF parsing
- Text chunking
- Embeddings
- ChromaDB vector storage
- Query endpoint
- LLM answer generation

---

# Missing / Needed Features

## 1. Better Project Structure

Current files are too mixed together.

Need simple separation between:
- routes
- controllers
- services

Suggested structure:

```text
app/
├── api/
├── controllers/
├── services/
└── main.py
```

---

## 2. Upload API

Currently ingestion is manual.

Need endpoint:

```http
POST /api/upload
```

Responsibilities:
- upload PDF files
- validate files
- parse documents
- chunk text
- store embeddings

---

## 3. Better Metadata

Current metadata is limited.

Need:

```text
page_number
chunk_index
total_chunks
upload_time
```

---

## 4. Better Retrieval

Current retrieval is only basic similarity search.

Suggested improvements:
- metadata filtering
- score threshold
- reranking

---

## 5. Error Handling

Need proper handling for:
- invalid files
- empty vector DB
- embedding failures
- retrieval failures
- missing API keys

---

# Recommended Order

## Phase 1
- Refactor project structure
- Add upload API
- Improve metadata

## Phase 2
- Improve retrieval
- Add filtering/reranking
- Add better error handling

---

# Final Note

The core RAG pipeline already works.

The main missing part is improving the software structure and making the system cleaner and closer to the project requirements.

