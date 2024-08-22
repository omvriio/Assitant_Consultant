from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, Any, TypedDict, List, Literal, Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio
from functools import lru_cache
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.tools import tool
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages


load_dotenv()

# Load the Nvidia API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

class DataState(TypedDict):
    df: pd.DataFrame

@tool
def find_file(filename: str) -> Dict[str, Any]:
    """This tool finds the full path of a file given its name."""
    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        if filename in files:
            return {"file_path": os.path.join(root, filename)}
    return {"error": f"File '{filename}' not found in the current directory or its subdirectories."}

@tool
def load_data(file_path: str) -> Dict[str, Any]:
    """This tool loads data from an Excel file."""
    try:
        print("Initializing data State")
        # s = DataState()
        print(f"Attempting to load file from: {file_path}")
        df = pd.read_excel(file_path)
        print(df.head())
        return {"data": df}
    except Exception as e:
        return {"error": f"Error loading file: {str(e)}"}

def analyze_row(row: pd.Series) -> Dict[str, Any]:
    context = f"""
    Analyze this row from a conformity matrix:
    Supplier Status: {row['Status SUPPLIER XYZ']}
    Stakeholder Status: {row['Status Stakeholder']}
    Supplier Comments: {row['Comments SUPPLIER XYZ']}
    Stakeholder Comments: {row['Comments Stakeholder']}

    Determine which of the following cases applies and provide a brief explanation:

    1. Supplier - Requirement Accepted
    2. Supplier - Requirement Accepted with Deviation
    3. Supplier - Requirement Rejected / Not Applicable
    4. Supplier - No Status
    5. Status Reset
    6. Supplier Changes Previously Accepted Status
    7. Supplier Requests Clarifications
    8. Supplier Requests Expert Meeting

    Format your response as follows:
    Thought process: [Your chain of reasoning]
    Case(s): [Case number(s)]
    Explanation: [Brief explanation]
    Confidence: [Score from 0 to 100]
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI consultant assistant analyzing a conformity matrix, identify rows to update."),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"input": context})

    return {
        "row": row.name + 2,
        "analysis": analysis
    }

@tool
def analyze_conformity_matrix(df: Dict[str, Any]) -> Dict[str, Any]:
    """This tool identifies rows that require updates after analyzing matrix."""
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {executor.submit(analyze_row, row): row for _, row in df.iterrows()}
        results = []
        for future in asyncio.as_completed(future_to_row):
            result = future.result()
            if any(f"Case {i}" in result['analysis'] for i in range(1, 9)):
                results.append(result)

    return {"rows_to_update": results}

@tool
def draft_email(df: Dict[str, Any], rows_to_update: List[Dict[str, Any]]) -> Dict[str, Any]:
    """This tool drafts an email based on the analysis of a conformity matrix row"""
    df = df["data"]
    
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
        [Body]
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

# Define tools
tools = [
    find_file,
    load_data,
    analyze_conformity_matrix,
    draft_email
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Now you can use llm_with_tools in your application
# load_dotenv()

# # Load the Nvidia API key
# os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
# llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

    

# @tool
# def find_file(filename: str) -> Dict[str, Any]:
#     """This tool finds the full path of a file given its name."""
#     current_dir = os.getcwd()
#     for root, dirs, files in os.walk(current_dir):
#         if filename in files:
#             return {"file_path": os.path.join(root, filename)}
#     return {"error": f"File '{filename}' not found in the current directory or its subdirectories."}
# @tool
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
# @lru_cache(maxsize=128)
# def get_llm_response(context: str) -> str:
#     # This function can be memoized to avoid redundant API calls
#     prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an AI consultant assistant analyzing a conformity matrix, identify rows to update."),
#             ("human", "{input}")
#         ])

#     chain = prompt | llm | StrOutputParser()
#     return chain.invoke({"input": context})

# async def analyze_row(row: pd.Series) -> Dict[str, Any]:
#     context = f"""
#         Analyze this row from a conformity matrix:
#         Supplier Status: {row['Status SUPPLIER XYZ']}
#         Stakeholder Status: {row['Status Stakeholder']}
#         Supplier Comments: {row['Comments SUPPLIER XYZ']}
#         Stakeholder Comments: {row['Comments Stakeholder']}

#         Determine which of the following cases applies and provide a brief explanation:

#         1. Supplier - Requirement Accepted:
#           Condition: Supplier accepts the requirement with no comments or conditions.

#         2. Supplier - Requirement Accepted with Deviation:
#           Condition: Supplier accepts the requirement with deviations.

#         3. Supplier - Requirement Rejected / Not Applicable:
#           Condition: Supplier rejects the requirement or marks it as not applicable.

#         4. Supplier - No Status:
#           Condition: Supplier has not provided a status for the requirement.

#         5. Status Reset:
#           Condition: Specification/package has been updated, impacting certain requirements.

#         6. Supplier Changes Previously Accepted Status:
#           Condition: Supplier changes the status of a previously accepted requirement.

#         7. Supplier Requests Clarifications:
#           Condition: Supplier asks for clarifications on requirements.

#         8. Supplier Requests Expert Meeting:
#           Condition: Supplier requests a meeting with experts.

#         Analyze the matrix data provided and respond with:
#         1. A concise chain of thoughts explaining your reasoning process.
#         2. The most applicable case number(s). Consider that some rows might have two cases combined.
#         3. A brief explanation of why this case (or cases) applies.
#         4. A confidence score from 0 to 100 for your conclusion.

#         Format your response as follows:
#         Thought process: [Your chain of reasoning]
#         Case(s): [Case number(s)]
#         Explanation: [Brief explanation]
#         Confidence: [Score from 0 to 100]

#         Remember, it's possible for multiple cases to apply to a single row. If you identify such a situation, 
#         include all relevant case numbers and explain the combination.
#         Make sure to make your answer as short and concise as possible.
#         """

        
#     analysis = await asyncio.to_thread(get_llm_response, context)
#     return {"row": row.name + 2, "analysis": analysis}

# async def analyze_conformity_matrix(df: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """This tool identifies rows that require updates after analyzing matrix."""
#     df = df["data"]

#     tasks = [analyze_row(row) for _, row in df.iterrows()]
#     results = await asyncio.gather(*tasks)
#     rows_to_update = [result for result in results if any(f"Case {i}" in result['analysis'] for i in range(1, 9))]
#     return {"rows_to_update": rows_to_update}


# @tool
# def draft_email(df: List[Dict[str, Any]], rows_to_update: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """This tool drafts an email based on the analysis of a conformity matrix row"""
#     df = df["data"]
    
#     for row_update in rows_to_update:
#         row_index = row_update['row'] - 2  
#         row_data = df.iloc[row_index]
        
#         context = f"""
#         Based on the following analysis of a conformity matrix row, draft an email:
#         {row_update['analysis']}

#         Use the following information from the conformity matrix to tailor the email:
#         Supplier Status: {row_data['Status SUPPLIER XYZ']}
#         Stakeholder Status: {row_data['Status Stakeholder']}
#         Supplier Comments: {row_data['Comments SUPPLIER XYZ']}
#         Stakeholder Comments: {row_data['Comments Stakeholder']}

#         Draft a professional and concise email addressing all the identified situations. The email should:
#         1. Acknowledge the supplier's response.
#         2. Address each identified case separately.
#         3. Provide clear next steps or requests for each case.
#         4. Maintain a cordial and professional tone throughout.

#         Use the following structure for the email:
#         Subject:
#         To:
#         Dear [Recipient],
#         opening
#         main_body
#         action
#         closing
        
#         Best regards,
#         [Your Name]

#         If the analysis doesn't indicate any cases that require action (such as case 1 where the requirement is simply accepted), 
#         return 'No email required' instead of drafting an email.
#         Make sure the email is concise and straight to the point.
#         """

#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an AI consultant assistant drafting an email based on the provided context."),
#             ("human", "{input}")
#         ])
        
#         chain = prompt | llm | StrOutputParser()
        
#         email_draft = chain.invoke({"input": context})
        
#         if email_draft.strip().lower() != 'no email required':
#             row_update['email_draft'] = email_draft

#     return {"email_drafts": rows_to_update}


# # Define tools
# tools = [
#     find_file,
#     load_data,
#     analyze_conformity_matrix,
#     draft_email
# ]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        """
        Initialize the Assistant with a runnable object.

        Args:
            runnable (Runnable): The runnable instance to invoke.
        """
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        """
        Call method to invoke the LLM and handle its responses.
        Re-prompt the assistant if the response is not a tool call or meaningful text.

        Args:
            state (State): The current state containing messages.
            config (RunnableConfig): The configuration for the runnable.

        Returns:
            dict: The final state containing the updated messages.
        """
        while True:
            result = self.runnable.invoke(state)  # Invoke the LLM
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Create the primary assistant prompt template
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant tasked with helping consultant assistants in their work or new Consultants in their tasks. "
            "You have access to tools specific for handling conformity matrix and drafting emails"
        ),
        ("placeholder", "{messages}"),
    ]
)

# Prompt our LLM and bind tools
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state: State) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

# Graph
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

import uuid
def predict_react_agent_answer(example: Dict[str, Any]):
    """Use this for answer evaluation"""
    
    initial_state = State(messages=[
        HumanMessage(content=example["input"])
    ])
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    final_state = None
    step_count = 0
    for output in react_graph.stream(initial_state, config):
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        print("Current output:")
        print(output)
        
        
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
    
    return {
        "response": output["messages"][-1].content if "messages" in output else None,
        "messages": output.get("messages", []),
        "final_state": final_state
    }

# Example usage
example = {
    "input": "Let's analyze conformity matrix data from the file 'exhibit_01_drones.xlsx' and draft emails if necessary.",
}
response = predict_react_agent_answer(example)