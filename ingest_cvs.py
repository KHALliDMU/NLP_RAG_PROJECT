import os
import re
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CV_FOLDER = "CVs"
VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "cv_collection"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RESET_VECTOR_DB = True


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted CV text.
    """

    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[_]{2,}", " ", text)
    text = re.sub(r"[-]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_cv_documents():
    """
    Parse raw CV PDFs and return LangChain documents with metadata.
    """

    all_documents = []

    for file_name in os.listdir(CV_FOLDER):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(CV_FOLDER, file_name)

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            for doc in documents:
                doc.page_content = clean_text(doc.page_content)

                # Skip only completely empty pages
                if not doc.page_content.strip():
                    continue

                doc.metadata["file_name"] = file_name
                doc.metadata["category"] = "CV"
                doc.metadata["source_path"] = file_path
                doc.metadata["char_count"] = len(doc.page_content)
                doc.metadata["parser"] = "PyPDFLoader"

                all_documents.append(doc)

    print(f"Loaded {len(all_documents)} valid pages from CV PDFs")
    return all_documents


def split_documents(documents):
    """
    Split CV text into chunks for better semantic retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP

    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vector_database(chunks):
    """
    Convert chunks into embeddings and store them in ChromaDB.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print("Vector database created successfully")
    return vector_db


def main():
    if RESET_VECTOR_DB and os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("Old vector database deleted")

    documents = load_cv_documents()

    if not documents:
        raise ValueError("No valid text was extracted from the CV PDFs.")

    chunks = split_documents(documents)
    build_vector_database(chunks)


if __name__ == "__main__":
    main()