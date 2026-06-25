import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


# Load API Key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


if not api_key:
    raise Exception("API Key missing")


# Create AI Model

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)



def investigate_claim(claim_details):


    prompt = f"""

You are a Healthcare Claims Investigation AI Agent.

Analyze the claim details below.

Provide:

1. Fraud Risk Level (Low / Medium / High)

2. Denial Risk Factors

3. Missing Information

4. Investigation Findings

5. Recommended Action


Claim Details:

{claim_details}


Create a professional healthcare investigation report.

"""


    response = llm.invoke(prompt)


    return response.content



# Interactive Claim Input

print("\nHealthcare Claims Investigation AI Agent")
print("---------------------------------------")


claim_id = input("Enter Claim ID: ")

patient = input("Enter Patient Name: ")

procedure = input("Enter Procedure: ")

amount = input("Enter Claim Amount: ")

diagnosis = input("Enter Diagnosis: ")

authorization = input("Authorization Status: ")



claim_details = f"""

Claim ID: {claim_id}

Patient Name: {patient}

Procedure: {procedure}

Claim Amount: {amount}

Diagnosis: {diagnosis}

Authorization Status: {authorization}

"""


report = investigate_claim(claim_details)



print("\nHealthcare Claims Investigation Report")
print("-------------------------------------")

print(report)