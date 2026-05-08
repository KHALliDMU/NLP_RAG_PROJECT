import os
from langchain_community.document_loaders import PyPDFLoader

CV_FOLDER = "CVs"

for file_name in os.listdir(CV_FOLDER):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(CV_FOLDER, file_name)

        print("\n" + "=" * 60)
        print("Testing:", file_name)
        print("=" * 60)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        full_text = " ".join([doc.page_content for doc in docs]).strip()

        if full_text:
            print("Text extracted")
            print("Characters:", len(full_text))
            print(full_text[:400])
        else:
            print(" No text extracted")