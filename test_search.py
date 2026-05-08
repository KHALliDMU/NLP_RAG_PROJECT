from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "cv_collection"


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def search_cvs(query, top_k=5):
    vector_db = load_vector_db()

    results = vector_db.similarity_search_with_score(
        query,
        k=top_k
    )

    for doc, score in results:
        print("\n" + "-" * 60)
        print("File:", doc.metadata.get("file_name"))
        print("Page:", doc.metadata.get("page"))
        print("Chunk:", doc.metadata.get("chunk_id"))
        print("Score:", score)
        print("Content:", doc.page_content[:500])


if __name__ == "__main__":
    search_cvs("Find candidates with Python and SQL experiencet", top_k=3)