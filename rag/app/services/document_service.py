"""Document service — handles PDF parsing and chunking."""
from __future__ import annotations

import re
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentService:
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted CV text."""
        text = text.replace("\x00", " ")
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[_]{2,}", " ", text)
        text = re.sub(r"[-]{3,}", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def parse_pdf(file_path: str) -> list[Any]:
        """Parse PDF and return documents with page metadata."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            for doc in documents:
                doc.page_content = DocumentService.clean_text(doc.page_content)
                if not doc.page_content.strip():
                    continue
                doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1

            # Filter out empty documents
            documents = [d for d in documents if d.page_content.strip()]
            return documents
        except Exception as e:
            raise ValueError(f"Failed to parse PDF {file_path}: {str(e)}")

    @staticmethod
    def chunk_documents(documents: list[Any]) -> list[Any]:
        """Split documents into chunks."""
        if not documents:
            raise ValueError("No documents to chunk")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DocumentService.CHUNK_SIZE,
            chunk_overlap=DocumentService.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_size"] = DocumentService.CHUNK_SIZE
            chunk.metadata["chunk_overlap"] = DocumentService.CHUNK_OVERLAP

        return chunks

    @staticmethod
    def validate_pdf(file_path: str) -> bool:
        """Validate that file is a valid PDF."""
        if not file_path.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF")
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b"%PDF":
                    raise ValueError("File is not a valid PDF")
            return True
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
