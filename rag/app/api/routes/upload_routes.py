"""Upload routes."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from pydantic import BaseModel

from app.controllers.upload_controller import UploadController
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api", tags=["Upload"])

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


def get_rag_service() -> RAGService:
    """Get RAG service from app context."""
    from app.main import _rag_service
    if _rag_service is None:
        raise RuntimeError("RAG service not initialized")
    return _rag_service


class UploadResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    chunks_added: int
    upload_time: str


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile, rag_service: RAGService = Depends(get_rag_service)):
    """
    Upload and process a PDF file.

    - Validates the file is a PDF
    - Parses the document
    - Chunks the text
    - Stores embeddings in ChromaDB
    - Saves file metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        # Validate file size
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")

        # Process the file
        controller = UploadController(rag_service)
        result = controller.upload_pdf(temp_path, original_filename=file.filename)

        # Save file to uploads directory
        dest_path = UPLOADS_DIR / file.filename
        shutil.copy(temp_path, dest_path)

        return UploadResponse(
            success=result["success"],
            message=result["message"],
            file_name=result["file_name"],
            chunks_added=result["chunks_added"],
            upload_time=result["upload_time"],
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
