import os
from typing import Dict, Any, TypedDict, List, Literal
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.tools import tool
load_dotenv()
class State(TypedDict):
    file_path: str
    df: pd.DataFrame
    rows_to_update: List[dict]
    query: str
    relevant_info: str
    output: str
    email_draft: str
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
def find_file(filename: str) -> Dict[str, Any]:
    """This tool finds the full path of a file given its name."""
    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        if filename in files:
            return {"file_path": os.path.join(root, filename)}
    return {"error": f"File '{filename}' not found in the current directory or its subdirectories."}

@tool
def analyze_conformity_matrix(file_path:str) -> Dict[str, Any]:
    """This tool identifies rows that require updates after analyzing matrix."""
    rows_to_update = []
    try:
        print(f"Attempting to load file from: {file_path}")
        df = pd.read_excel(file_path, nrows=2)
        print(df)
    except Exception as e:
        return {"error": f"Error loading file: {str(e)}"}

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
        Make sure to make your answer as short and concise as possible.
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

    return {"rows_to_update": rows_to_update}

# analyze_matrix_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI consultant assistant analyzing a conformity matrix."),
#     ("human", "Please analyze the following conformity matrix data: {df}")
# ])

# analyze_matrix_chain = analyze_matrix_prompt | llm | StrOutputParser() | analyze_conformity_matrix

# analyze_matrix_tool = analyze_matrix_chain.as_tool(
#     name="analyze_conformity_matrix",
#     description="Analyze a conformity matrix to identify rows that require updates",
# )

def draft_email(df: List[Dict[str, Any]], rows_to_update: List[Dict[str, Any]]) -> Dict[str, Any]:
    """This tool drafts an email based on the analysis of a conformity matrix row"""
    df = pd.DataFrame(df)
    
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

    return {"email_drafts": rows_to_update}

draft_email_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI consultant assistant drafting emails based on conformity matrix analysis."),
    ("human", "Please draft emails for the following rows that need updates: {rows_to_update}. Use the data from the conformity matrix: {df}")
])

draft_email_chain = draft_email_prompt | llm | StrOutputParser() | draft_email

draft_email_tool = draft_email_chain.as_tool(
    name="draft_email",
    description="Draft an email based on the analysis of a conformity matrix row",
)

# Define tools
tools = [
    find_file,
    analyze_conformity_matrix,
    draft_email_tool
]

tool_node = ToolNode(tools)

llm_ = llm.bind_tools(tools)

# In this graph continue until no more tools
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# Call the model on the current messages
def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_.invoke(messages)
    return {"messages": messages + [response]}

workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Define edges of the graph
workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": END
    }
)

workflow.add_edge("tools", "agent")

# Check structure of graph by compiling it
app = workflow.compile()

from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))
# Test the agent graph
# def test_agent_graph():
#     initial_state = {
#         "messages": [
#             {
#                 "role": "human",
#                 "content": "Please load the data from 'exhibit_01_drones.xlsx', analyze it, and draft emails for any rows that need updates."
#             }
#         ]
#     }
    
#     for output in app.stream(initial_state):
#         if "__end__" in output:
#             final_state = output["__end__"]
#             break
    
#     # Print the final state or relevant information
#     print("Final State:")
#     print(final_state)

# # Run the test
# if __name__ == "__main__":
#     print("Please load the data from 'exhibit_01_drones.xlsx', analyze it, and draft emails for any rows that need updates.")
#     test_agent_graph()

def test_agent_graph():
    initial_state = State(messages=[
            HumanMessage(content="Let's analyze conformity matrix data from the file 'exhibit_01_drones.xlsx' and draft emails if necessary.")],
            filename = "exhibit_01_drones.xlsx"
    )
    
    final_state = None
    step_count = 0
    for output in app.stream(initial_state):
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        print("Current output:")
        print(output)
        
        # Print the current state
        print("\nCurrent State:")
        for key, value in output.items():
            if key != "messages":
                print(f"{key}: {value}")
        
        # Print messages separately for better readability
        if "messages" in output:
            print("\nMessages:")
            for msg in output["messages"]:
                print(f"Role: {msg.type}")
                print(f"Content: {msg.content}")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"Tool Calls: {msg.tool_calls}")
                print("---")
        
        if "__end__" in output:
            final_state = output["__end__"]
            break
    
    if final_state:
        print("\nFinal State:")
        print(final_state)
    else:
        print("\nWarning: The workflow did not reach a final state.")
        print("Last output:")
        print(output)
    

# Run the test
if __name__ == "__main__":
    print("Starting the test...")
    print("Please load the data from './exhi.xlsx', analyze it, and draft emails for any rows that need updates.")
    test_agent_graph()