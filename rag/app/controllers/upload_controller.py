"""Upload controller — handles document upload business logic."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.services.document_service import DocumentService
from app.services.rag_service import RAGService


class UploadController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.document_service = DocumentService()

    def upload_pdf(self, file_path: str) -> dict:
        """Upload and process a PDF file."""
        try:
            # Validate file
            self.document_service.validate_pdf(file_path)

            # Extract file name
            file_name = Path(file_path).name

            # Parse PDF
            documents = self.document_service.parse_pdf(file_path)

            if not documents:
                raise ValueError(f"No content extracted from {file_name}")

            # Chunk documents
            chunks = self.document_service.chunk_documents(documents)

            # Add to vector DB
            upload_time = datetime.now().isoformat()
            result = self.rag_service.add_documents(chunks, file_name, upload_time)

            return {
                "success": True,
                "message": f"Successfully uploaded {file_name}",
                "file_name": file_name,
                "chunks_added": result["chunks_added"],
                "upload_time": result["upload_time"],
            }
        except ValueError as e:
            raise ValueError(f"Validation error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Upload failed: {str(e)}")
