import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# Load API key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise Exception("OPENAI_API_KEY not found")


print("API Key Loaded Successfully")


# Documents folder

DOCUMENT_PATH = "Documents"


documents = []

for file_name in os.listdir(DOCUMENT_PATH):

    file_path = os.path.join(
        DOCUMENT_PATH,
        file_name
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()

        documents.append(
            {
                "content": content,
                "source": file_name
            }
        )


print("Total Documents Loaded:", len(documents))


# Split documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


chunks = []
metadatas = []


for doc in documents:

    split_texts = text_splitter.split_text(
        doc["content"]
    )

    for text in split_texts:

        chunks.append(text)

        metadatas.append(
            {
                "source": doc["source"]
            }
        )


print("Chunks ready:", len(chunks))


# Create embeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)


# Create Vector DB

vector_db = Chroma.from_texts(
    texts=chunks,
    metadatas=metadatas,
    embedding=embeddings,
    persist_directory="Vector_DB"
)


print("Vector Database Created Successfully")