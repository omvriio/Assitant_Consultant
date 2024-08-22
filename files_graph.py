from typing import Dict, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import fitz  # PyMuPDF
import re
import os

class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_file: str
    processed_files: List[str]
    chunks: List[Dict]

initial_state = AgentState(
    messages=[HumanMessage(content="Let's process some PDF files.")],
    current_file="",
    processed_files=[],
    chunks=[]
)

def process_pdf(state: AgentState) -> AgentState:
    if not state["current_file"]:
        return state

    # Create a directory for saving images
    pdf_name = os.path.splitext(os.path.basename(state["current_file"]))[0]
    image_dir = f"images_{pdf_name}"
    os.makedirs(image_dir, exist_ok=True)

    doc = fitz.open(state["current_file"])
    chunks = []
    toc = []
    current_section = ""

    def add_chunk(content: str, chunk_type: str, metadata: Dict):
        chunks.append({
            "text": content,
            "type": chunk_type,
            "metadata": {**metadata, "section": current_section}
        })

    # Extract table of contents
    for i in doc.get_toc():
        toc.append(f"{'  ' * (i[0] - 1)}{i[1]}")
    
    if toc:
        add_chunk("\n".join(toc), "table_of_contents", {"page": 0})
    # print(toc)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        # Check for new section
        section_match = re.match(r'^\d+\s+(.+)$', text.strip().split('\n')[0])
        if section_match:
            current_section = section_match.group(1)

        # Process tables
        tables = page.find_tables()
        for table in tables:
            table_text = "\n".join([" | ".join(str(cell) if cell is not None else "" for cell in row) for row in table.extract()])
            add_chunk(table_text, "table", {"page": page_num + 1})

        # Process images
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image to file
            image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(image_dir, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            # Find image caption
            caption = ""
            caption_match = re.search(r'Figure \d+\s*:\s*(.+)', text)
            if caption_match:
                caption = caption_match.group(1)

            add_chunk(f"Image: {caption}", "image", {
                "page": page_num + 1,
                "image_index": img_index,
                "image_path": image_path
            })

        # Process text (excluding tables and image captions)
        text_chunks = text.split('\n\n')
        for chunk in text_chunks:
            if not any(chunk.startswith(f"{i}") for i in range(1, 10)) and "Figure" not in chunk:
                add_chunk(chunk, "text", {"page": page_num + 1})

    state["chunks"].extend(chunks)
    state["processed_files"].append(state["current_file"])
    state["current_file"] = ""
    
    doc.close()
    print("State after process_pdf:--------------------------------------------------------")
    print(f"Current file: {state['current_file']}")
    print(f"Processed files: {state['processed_files']}")
    print(f"Number of chunks: {len(state['chunks'])}")
    return state

def store_in_chroma(state: AgentState) -> AgentState:
    # Use OllamaEmbeddings with the mxbai model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    texts = [chunk["text"] for chunk in state["chunks"]]
    metadatas = [chunk["metadata"] for chunk in state["chunks"]]
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings
    )
    
    # Clear the chunks after storing
    state["chunks"] = []
    
    print("State after store_in_chroma:----------------------------------------------------")
    print(f"Current file: {state['current_file']}")
    print(f"Processed files: {state['processed_files']}")
    print(f"Number of chunks: {len(state['chunks'])}")
    return state

workflow = StateGraph(AgentState)

workflow.add_node("process_pdf", process_pdf)
workflow.add_node("store_in_chroma", store_in_chroma)

workflow.add_edge("process_pdf", "store_in_chroma")
workflow.add_edge("store_in_chroma", END)

workflow.set_entry_point("process_pdf")

def run_workflow(file_path: str):
    app = workflow.compile()
    
    state = initial_state.copy()
    state["current_file"] = file_path
    
    print("Initial state:------------------------------------------------------------------")  
    print(f"Current file: {state['current_file']}")
    print(f"Processed files: {state['processed_files']}")
    print(f"Number of chunks: {len(state['chunks'])}")
    
    for output in app.stream(state):
        if output.get("type") == "end":
            print("Workflow completed.")
            break
        # else:
        #     print(f"Current step: {output}")

# Example usage
run_workflow("data\STA14_STL_ETI_Risk_Assessment_Methodology_01552_13_01824.V5.1_JUNE_2023-1-5.pdf")