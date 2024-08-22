import os
import streamlit as st
from typing import Dict, Any
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import Graph, StateGraph, END
from langchain.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()

# Load the Nvidia API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
# Define the state structure
class State(TypedDict):
    file_path: str
    df: pd.DataFrame
    rows_to_update: List[dict]
    query: str
    relevant_info: str
    output: str
    email_draft: str
    
@tool
def analyze_conformity_matrix(state: State) -> State:
    df = state['df']
    llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
    rows_to_update = []

    for index, row in df.iterrows():
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
        Make sure to make you answer as short and concise as possible.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI consultant assistant analyzing a conformity matrix, identify rows to update."),
            ("human", "{input}")
        ])

        chain = prompt | llm | StrOutputParser()

        analysis = chain.invoke({"input": context})

        if any(f"Case {i}" in analysis for i in range(1, 9)):
            rows_to_update.append({
                "row": index + 2,
                "analysis": analysis
            })

    state["rows_to_update"] = rows_to_update
    return state

def load_data(state: State) -> State:
    file_path = state['file_path']
    state['df'] = pd.read_excel(file_path,nrows=2)
    return state

@tool
def draft_email(state: State) -> State:
    llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
    
    for row_update in state['rows_to_update']:
        row_index = row_update['row'] - 2  
        row_data = state['df'].iloc[row_index]
        
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

    return state

tools = [draft_email, analyze_conformity_matrix]
tool_node = ToolNode(tools)

llm_ =llm.bind_tools(tools)



def retriever(state: State) -> State:
    # Implement retriever logic here if needed
    state["relevant_info"] = "Retrieved relevant information"
    return state

def generate_output(state: State) -> State:
    # Implement output generation logic here if needed
    state["output"] = "Generated output based on relevant information"
    return state

# Define the graph
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("retriever", retriever)
workflow.add_node("generate_output", generate_output)
workflow.add_node("load_data", load_data)
workflow.add_node("analyze_matrix", analyze_conformity_matrix)
workflow.add_node("draft_email", draft_email)

# Connect the nodes
workflow.add_edge("retriever", "generate_output")
workflow.add_edge("generate_output", "load_data")
workflow.add_edge("load_data", "analyze_matrix")
workflow.add_edge("analyze_matrix", "draft_email")

# Set the entry point
workflow.set_entry_point("retriever")

# Set the exit point
workflow.set_finish_point("draft_email")

# Compile the graph
app = workflow.compile()

# Streamlit app
def main():
    st.title("Conformity Matrix Analysis and Email Drafting Tool")

    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        file_path = "temp.xlsx"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        

        if st.button("Analyze and Draft Email"):
            initial_state = State(
                file_path=file_path,
                df=pd.DataFrame(),
                rows_to_update=[],
                query="",
                relevant_info="",
                output="",
                email_draft="",

            )
            
            final_state = app.invoke(initial_state)
            
            st.subheader("Analysis Results")
            for update in final_state["rows_to_update"]:
                st.write(f"Row {update['row']}:")
                st.write(update['analysis'])
                if 'email_draft' in update:
                    st.subheader(f"Generated Email for Row {update['row']}")
                    st.write(update['email_draft'])
                st.write("-" * 50)

        # Clean up the temporary file
        os.remove(file_path)

if __name__ == "__main__":
    main()