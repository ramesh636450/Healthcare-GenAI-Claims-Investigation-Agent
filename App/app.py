import os
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


# Load API Key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# Page Setup

st.set_page_config(
    page_title="Healthcare Claims AI Agent",
    layout="wide"
)


st.title("🏥 Healthcare Claims Investigation AI Agent")

st.write(
    "AI-powered claims audit using RAG + Healthcare Guidelines"
)



# Load Claims Data

claims_df = pd.read_excel(
    "Healthcare_GenAI_Claims_Dataset.xlsx"
)



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



# Claim Selection

claim_id = st.selectbox(
    "Select Claim ID",
    claims_df["Claim_ID"]
)



if st.button("Analyze Claim"):


    claim = claims_df[
        claims_df["Claim_ID"] == claim_id
    ]


    claim_details = claim.to_string(
        index=False
    )


    docs = retriever.invoke(
        claim_details
    )


    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )


    prompt = f"""

You are a Healthcare Claims Investigation AI Agent.


Analyze this claim:

{claim_details}


Healthcare Guidelines:

{context}


Create an audit report:

1. Risk Assessment
2. Fraud Risk Level
3. Denial Risk Factors
4. Guidelines Used
5. Missing Information
6. Investigation Findings
7. Recommended Action

Separate fraud risk and denial risk.

"""


    response = llm.invoke(prompt)


    st.subheader(
        "Healthcare Investigation Report"
    )


    st.write(
        response.content
    )