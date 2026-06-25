from document_loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


documents = load_documents()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


all_chunks = []


for doc in documents:

    chunks = text_splitter.split_text(doc["content"])

    for chunk in chunks:

        all_chunks.append({
            "source": doc["file_name"],
            "text": chunk
        })


print("Total Chunks Created:", len(all_chunks))


for chunk in all_chunks[:5]:

    print("\nSource:", chunk["source"])
    print(chunk["text"])