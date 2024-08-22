import asyncio
from functools import lru_cache
import os
from typing import Dict, Any, TypedDict, List
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_nvidia_ai_endpoints import ChatNVIDIA
import pandas as pd
from dotenv import load_dotenv
import time
y =time.time()
load_dotenv()

# Load the Nvidia API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
class DataState(TypedDict):
    df: pd.DataFrame
# def load_data(file_path: str) -> Dict[str, Any]:
#     """This tool loads data from an Excel file."""
#     try:
#         print("initializint data State")
#         # s = DataState()
#         print(f"Attempting to load file from: {file_path}")
#         df = pd.read_excel(file_path, nrows=2)
#         print(df)
#         return {"data": df}
#     except Exception as e:
#         return {"error": f"Error loading file: {str(e)}"}
@lru_cache(maxsize=128)
def get_llm_response(context: str) -> str:
    # This function can be memoized to avoid redundant API calls
    prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI consultant assistant analyzing a conformity matrix, identify rows to update."),
            ("human", "{input}")
        ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": context})

async def analyze_row(row: pd.Series) -> Dict[str, Any]:
    context = f"""
        Analyze this row from a conformity matrix:
        Supplier Status: {row['Status SUPPLIER XYZ']}
        Stakeholder Status: {row['Status Stakeholder']}
        Supplier Comments: {row['Comments SUPPLIER XYZ']}
        Stakeholder Comments: {row['Comments Stakeholder']}

        Determine which of the following cases applies and provide a brief explanation:

        1. Supplier - Requirement Accepted:
          Condition: Supplier accepts the requirement with no comments or conditions.

        2. Supplier - Requirement Accepted with Deviation:
          Condition: Supplier accepts the requirement with deviations.

        3. Supplier - Requirement Rejected / Not Applicable:
          Condition: Supplier rejects the requirement or marks it as not applicable.

        4. Supplier - No Status:
          Condition: Supplier has not provided a status for the requirement.

        5. Status Reset:
          Condition: Specification/package has been updated, impacting certain requirements.

        6. Supplier Changes Previously Accepted Status:
          Condition: Supplier changes the status of a previously accepted requirement.

        7. Supplier Requests Clarifications:
          Condition: Supplier asks for clarifications on requirements.

        8. Supplier Requests Expert Meeting:
          Condition: Supplier requests a meeting with experts.

        Analyze the matrix data provided and respond with:
        1. A concise chain of thoughts explaining your reasoning process.
        2. The most applicable case number(s). Consider that some rows might have two cases combined.
        3. A brief explanation of why this case (or cases) applies.
        4. A confidence score from 0 to 100 for your conclusion.

        Format your response as follows:
        Thought process: [Your chain of reasoning]
        Case(s): [Case number(s)]
        Explanation: [Brief explanation]
        Confidence: [Score from 0 to 100]

        Remember, it's possible for multiple cases to apply to a single row. If you identify such a situation, 
        include all relevant case numbers and explain the combination.
        Make sure to make your answer as short and concise as possible.
        """

        
    analysis = await asyncio.to_thread(get_llm_response, context)
    return {"row": row.name + 2, "analysis": analysis}

async def analyze_conformity_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """This tool identifies rows that require updates after analyzing matrix."""
    tasks = [analyze_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    rows_to_update = [result for result in results if any(f"Case {i}" in result['analysis'] for i in range(1, 9))]
    return {"rows_to_update": rows_to_update}

# Usage
file_path = "./processed_data/exhibit_01_drones.xlsx"
# df = load_data(file_path)["data"]
df=pd.read_excel(file_path,nrows=10)
rows_to_update = asyncio.run(analyze_conformity_matrix(df))

print(rows_to_update)
def draft_email(file_path: str, rows_to_update: List[Dict[str, Any]]) -> Dict[str, Any]:
    """This tool drafts an email based on the analysis of a conformity matrix row"""
    df = pd.read_excel(file_path, nrows=10)

    
    for row_update in rows_to_update:
        row_index = row_update['row'] - 2  
        row_data = df.iloc[row_index]
        
        context = f"""
        Based on the following analysis of a conformity matrix row, draft an email:
        {row_update['analysis']}

        Use the following information from the conformity matrix to tailor the email:
        Supplier Status: {row_data['Status SUPPLIER XYZ']}
        Stakeholder Status: {row_data['Status Stakeholder']}
        Supplier Comments: {row_data['Comments SUPPLIER XYZ']}
        Stakeholder Comments: {row_data['Comments Stakeholder']}

        Draft a professional and concise email addressing all the identified situations. The email should:
        1. Acknowledge the supplier's response.
        2. Address each identified case separately.
        3. Provide clear next steps or requests for each case.
        4. Maintain a cordial and professional tone throughout.

        Use the following structure for the email:
        Subject:
        To:
        Dear [Recipient],
        opening
        main_body
        action
        closing
        
        Best regards,
        [Your Name]

        If the analysis doesn't indicate any cases that require action (such as case 1 where the requirement is simply accepted), 
        return 'No email required' instead of drafting an email.
        Make sure the email is concise and straight to the point.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI consultant assistant drafting an email based on the provided context."),
            ("human", "{input}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        email_draft = chain.invoke({"input": context})
        
        if email_draft.strip().lower() != 'no email required':
            row_update['email_draft'] = email_draft
            print(row_update['email_draft'])
    return {"email_drafts": rows_to_update}
draft_email(file_path, rows_to_update)
x = time.time()
print(x-y)