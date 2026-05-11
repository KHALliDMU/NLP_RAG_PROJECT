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


def search_cvs(query, top_k=3):
    vector_db = load_vector_db()

    results = vector_db.similarity_search_with_score(
        query,
        k=top_k
    )

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    for rank, (doc, score) in enumerate(results, start=1):

        print("\n" + "-" * 60)
        print(f"Rank: {rank}")
        print("File:", doc.metadata.get("file_name"))
        print("Page:", doc.metadata.get("page"))
        print("Chunk:", doc.metadata.get("chunk_id"))
        print("Chunk Size:", doc.metadata.get("chunk_size"))
        print("Overlap:", doc.metadata.get("chunk_overlap"))
        print("Score:", score)

        print("\nContent:")
        print(doc.page_content[:500])


if __name__ == "__main__":

    queries = [
        "Find candidates with Python and SQL experience",
        "Find candidates with machine learning projects",
        "Find candidates with frontend development skills"
    ]

    for q in queries:
        search_cvs(q)