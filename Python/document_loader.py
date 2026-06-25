import os


DOCUMENT_PATH = "Documents"


def load_documents():

    documents = []

    for file_name in os.listdir(DOCUMENT_PATH):

        if file_name.endswith(".txt"):

            file_path = os.path.join(DOCUMENT_PATH, file_name)

            with open(file_path, "r", encoding="utf-8") as file:

                content = file.read()

                documents.append({
                    "file_name": file_name,
                    "content": content
                })

    return documents


docs = load_documents()


print("Total Documents Loaded:", len(docs))


for doc in docs:
    print("\nDocument:", doc["file_name"])
    print(doc["content"][:200])