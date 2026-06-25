import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# Load API Key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


if not api_key:
    raise Exception("API Key missing")


# Load Vector Database

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)


vector_db = Chroma(
    persist_directory="Vector_DB",
    embedding_function=embeddings
)


# Retriever

retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# GPT Model

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)



def ask_question(question):

    # Search documents

    documents = retriever.invoke(question)


    context = "\n\n".join(
        [
            doc.page_content
            for doc in documents
        ]
    )


    # Capture sources

    sources = list(
        set(
            [
                doc.metadata.get(
                    "source",
                    "Unknown"
                )
                for doc in documents
            ]
        )
    )


    prompt = f"""

You are a Healthcare Claims Investigation AI Assistant.

Answer the question using only the healthcare knowledge provided.

Healthcare Context:

{context}


Question:

{question}


Provide:
1. Clear explanation
2. Important points
3. Practical healthcare claims perspective

"""


    response = llm.invoke(prompt)


    return response.content, sources



# User Input

question = input(
    "\nAsk Healthcare Question: "
)


answer, sources = ask_question(question)



print("\nAI Assistant Response:")
print(answer)


print("\nSources Used:")

for source in sources:
    print("-", source)