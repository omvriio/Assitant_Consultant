import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document as DOCC
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# Load the Nvidia API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def process_excel_and_embed(file_path, collection_name):
    df = pd.read_excel(file_path)
    
    combined_texts = []
    for index, row in df.iterrows():
        text = f"Row {index + 2}:\n"
        text += f"Supplier Status: {row['Status SUPPLIER XYZ']}\n"
        text += f"Stakeholder Status: {row['Status Stakeholder']}\n"
        text += f"Supplier Comments: {row['Comments SUPPLIER XYZ']}\n"
        text += f"Stakeholder Comments: {row['Comments Stakeholder']}\n"
        
        combined_texts.append({
            'text': text,
            'metadata': {'row': index + 2}
        })
    
    # Create Document objects
    docs = [DOCC(page_content=item['text'], metadata=item['metadata']) for item in combined_texts]
    
    # Split documents
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3400, chunk_overlap=20, separator="\n")
    doc_splits = text_splitter.split_documents(docs)
    
    # Initialize Chroma with OLLAMA embeddings
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        collection_name=collection_name
    )
    
    # Add documents to Chroma
    db.add_documents(doc_splits)
    
    print(f"Data has been ingested into vector database. Collection: {collection_name}")
    return db

# def retrieve_cases(db, llm):
#     # Define the cases
#     cases = [
#          "1. Supplier - Requirement Accepted: Condition: Supplier accepts the requirement with no comments or conditions.",
#         "2. Supplier - Requirement Accepted with Deviation: Condition: Supplier accepts the requirement with deviations.",
#         "3. Supplier - Requirement Rejected / Not Applicable: Condition: Supplier rejects the requirement or marks it as not applicable.",
#         "4. Supplier - No Status: Condition: Supplier has not provided a status for the requirement.",
#         "5. Status Reset: Condition: Specification/package has been updated, impacting certain requirements.",
#         "6. Supplier Changes Previously Accepted Status:- Condition: Supplier changes the status of a previously accepted requirement.",
#         "7. Supplier Requests Clarifications: Condition: Supplier asks for clarifications on requirements.",
#         "8. Supplier Requests Expert Meeting: Condition: Supplier requests a meeting with experts.",
#     ]

#     results = []

#     for i, case in enumerate(cases, 1):
#         query = f"Find rows where {case}"
#         docs = db.similarity_search(query, k=5)  # Adjust k as needed
        
#         if docs:
#             context = f"""
#             Analyze the following rows from a conformity matrix for Case {i}: {case}

#             {' '.join([doc.page_content for doc in docs])}

#             For each relevant row:
#             1. Confirm if it truly matches Case {i}.
#             2. Provide a brief explanation of why it matches or doesn't match.
#             3. If it matches, suggest a concise email response addressing the situation.

#             Format your response for each row as follows:
#             Row: [Row number]
#             Matches Case {i}: [Yes/No]
#             Explanation: [Brief explanation]
#             Suggested Email: [Concise email draft or 'No email required' if it doesn't match]

#             Provide your analysis for all relevant rows in a single response.
#             """

#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", "You are an AI consultant assistant analyzing a conformity matrix and drafting emails."),
#                 ("human", "{input}")
#             ])

#             chain = prompt | llm | StrOutputParser()

#             try:
#                 analysis = chain.invoke({"input": context})
#                 results.append((i, case, analysis))
#             except Exception as e:
#                 print(f"Error analyzing Case {i}: {e}")

#     return results

# # Update the usage section
# file_path = 'exhibit_01_drones.xlsx'
# collection_name = "conformity_matrix"

# # Process Excel and embed data
# db = process_excel_and_embed(file_path, collection_name)

# # Initialize LLM
# llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

# # Retrieve and analyze cases
# case_results = retrieve_cases(db, llm)

# # Print results
# for case_num, case_desc, analysis in case_results:
#     print(f"\nCase {case_num}: {case_desc}")
#     print(analysis)
#     print("-" * 50)


# def retrieve_cases(db, llm):
#     # Define the cases
#     cases = [
#         "1. Supplier - Requirement Accepted: Condition: Supplier accepts the requirement with no comments or conditions.",
#         "2. Supplier - Requirement Accepted with Deviation: Condition: Supplier accepts the requirement with deviations.",
#         "3. Supplier - Requirement Rejected / Not Applicable: Condition: Supplier rejects the requirement or marks it as not applicable.",
#         "4. Supplier - No Status: Condition: Supplier has not provided a status for the requirement.",
#         "5. Status Reset: Condition: Specification/package has been updated, impacting certain requirements.",
#         "6. Supplier Changes Previously Accepted Status: Condition: Supplier changes the status of a previously accepted requirement.",
#         "7. Supplier Requests Clarifications: Condition: Supplier asks for clarifications on requirements.",
#         "8. Supplier Requests Expert Meeting: Condition: Supplier requests a meeting with experts."
#     ]

#     query = f"Find rows closed by the supplier"
#     docs = db.similarity_search(query, k=20)  # Adjust k as needed
    
#     if docs:
#         context = f"""
#         Analyze the following rows from a conformity matrix:

#         {' '.join([doc.page_content for doc in docs])}

#         Determine which of the following cases applies and provide a brief explanation:

#         {' '.join(cases)}

#         For each row:
#         1. A concise chain of thoughts explaining your reasoning process.
#         2. The most applicable case number(s). Consider that some rows might have two cases combined.
#         3. A brief explanation of why this case (or cases) applies.
#         4. A confidence score from 0 to 100 for your conclusion.

#         Format your response for each row as follows:
#         Row: [Row number]
#         Thought process: [Your chain of reasoning]
#         Case(s): [Case number(s)]
#         Explanation: [Brief explanation]
#         Confidence: [Score from 0 to 100]

#         Remember, it's possible for multiple cases to apply to a single row. If you identify such a situation, 
#         include all relevant case numbers and explain the combination.
#         """

#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an AI consultant assistant analyzing a conformity matrix and identifying cases."),
#             ("human", "{input}")
#         ])

#         chain = prompt | llm | StrOutputParser()

#         try:
#             analysis = chain.invoke({"input": context})
#         except Exception as e:
#             print(f"Error analyzing: {e}")

#     return analysis

# # Update the usage section
# file_path = 'exhibit_01_drones.xlsx'
# collection= "conformity_matrix"

# # Process Excel and embed data
# db = process_excel_and_embed(file_path, collection)
# #load the db from the previous run
# # db = Chroma(persist_directory="./chroma_db", embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"), collection_name=collection)

# # Initialize LLM
# llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

# # Retrieve and analyze cases
# case_results = retrieve_cases(db, llm)

# # Print results
# print(case_results)