import os
import pandas as pd

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


# Load API Key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise Exception("API Key missing")


# Load Claims Dataset

file_path = "Healthcare_GenAI_Claims_Dataset.xlsx"

claims_df = pd.read_excel(file_path)



# Load Vector Database

vector_db = Chroma(
    persist_directory="Vector_DB",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )
)


retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)



# AI Model

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)



print("\nHealthcare RAG Claims Investigation AI Agent")
print("--------------------------------------------")


print("\nAvailable Claims:")

print(
    claims_df[
        ["Claim_ID","Provider_Name","Claim_Amount"]
    ]
)



claim_id = input(
    "\nEnter Claim ID to Investigate: "
)



claim = claims_df[
    claims_df["Claim_ID"] == claim_id
]



if claim.empty:

    print("Claim not found")

    exit()



# Convert claim row

claim_details = claim.to_string(
    index=False
)



# Retrieve healthcare guidelines

docs = retriever.invoke(
    claim_details
)



context = "\n\n".join(
    [doc.page_content for doc in docs]
)



# Final AI Prompt

prompt = f"""

You are a Healthcare Claims Investigation AI Agent.

Analyze the claim using the healthcare guidelines provided.

CLAIM DETAILS:

{claim_details}


HEALTHCARE GUIDELINES:

{context}


Prepare an audit report with:

1. Claim Risk Assessment

2. Fraud Risk Level

3. Denial Risk Factors

4. Relevant Healthcare Guidelines

5. Missing Information

6. Investigation Findings

7. Recommended Action


Separate fraud risk and denial risk clearly.

"""


response = llm.invoke(prompt)



print("\nHealthcare Claims Investigation Report")
print("--------------------------------------")

print(response.content)