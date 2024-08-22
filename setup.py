from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from langchain_community.document_loaders import UnstructuredFileLoader as partition_pdf
from pathlib import Path
from typing import Callable, Dict
import pypandoc
import multiprocessing
from functools import partial
import os
import openpyxl
from typing import List, Dict, Generator
from openpyxl.utils.exceptions import InvalidFileException
import pandas as pd

def iter_block_items(parent):
    """
    Iterate through all paragraphs and tables in a document or cell
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")
    
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def doc_docx(input_path: str, output_format: str = 'docx') -> str:
    input_path = os.path.abspath(input_path)
    output_path = os.path.splitext(input_path)[0] + f".{output_format}"
    
    try:
        pypandoc.convert_file(input_path, output_format, outputfile=output_path)
        print(f"Converted {input_path} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return input_path

def process_word(file_path: str, max_rows_per_chunk: int = 10) -> list:
    file_path = os.path.abspath(file_path)
    
    if file_path.lower().endswith('.doc'):
        print(f"Converting {file_path} to .docx using pandoc")
        file_path = doc_docx(file_path, 'docx')
    print(f"Attempting to open file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    try:
        doc = Document(file_path)
    except Exception as e:
        print(f"Error opening document: {e}")
        return []

    chunks = []
    current_chunk = []
    current_heading = ""

    def add_chunk():
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk.clear()

    def process_table(table, title):
        if len(table.rows) <= max_rows_per_chunk + 1:  # +1 for header row
            table_text = [title]
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                table_text.append(' | '.join(row_text))
            chunks.append(f"[Table]\n" + '\n'.join(table_text))
        else:
            header_row = [cell.text for cell in table.rows[0].cells]
            header = ' | '.join(header_row)
            
            for i in range(1, len(table.rows), max_rows_per_chunk):
                table_chunk = [f"{title} (Part {i//max_rows_per_chunk + 1})", header]
                for row in table.rows[i:i+max_rows_per_chunk]:
                    row_text = [cell.text for cell in row.cells]
                    table_chunk.append(' | '.join(row_text))
                chunks.append(f"[Table]\n" + '\n'.join(table_chunk))

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            if block.style.name.startswith('Heading'):
                add_chunk()
                current_heading = f"[{block.style.name}] {block.text}"
                current_chunk.append(current_heading)
            else:
                current_chunk.append(block.text)
        elif isinstance(block, Table):
            add_chunk()
            process_table(block, current_heading)

    add_chunk()  # Add any remaining content

    # save images
    for rel in doc.part.rels.values():
        if rel.is_external:
            # Log the external link (the target)
            print(f"External link found: {rel.target_ref}")
        else:
            if "image" in rel.reltype:
                image_path = os.path.join(os.path.dirname(file_path), "images")
                os.makedirs(image_path, exist_ok=True)
                image_name = os.path.basename(rel.target_ref)
                image_save_path = os.path.join(image_path, image_name)
                with open(image_save_path, "wb") as image_file:
                    image_file.write(rel.target_part.blob)
                chunks.append(f"Image saved at {rel.target_ref}")

    output_path = file_path.rsplit('.', 1)[0] + "_chunks.txt"
    with open(f"processed_data/{os.path.basename(output_path)}", "w", encoding='utf-8') as file:
        for i, chunk in enumerate(chunks, 1):
            file.write(f"Chunk {i}:\n{chunk}\n")
            file.write("\n----------------------------------------------------------------------------------------\n")
    print(f"Processed Word: {file_path}. Chunks saved to {output_path}")

    return chunks
def process_pdf(file_path: str) -> None:
    #use unstructured partitioning to process the pdf
    chunks= partition_pdf(file_path, chunking_strategy="by_title", mode="elements")
    output_path = file_path.rsplit('.', 1)[0] + "_chunks.txt"
    with open(f"processed_data/{os.path.basename(output_path)}", "w", encoding='utf-8') as file:
        file.write(chunks.load())
    print(f"Processed PDF: {file_path}. Chunks saved to {output_path}")
    return chunks
    

import pandas as pd

def process_excel(file_path: str) -> None:
    file_path = os.path.abspath(file_path)
    output_path = file_path.rsplit('.', 1)[0] + "_chunks.txt"
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    with open(f"processed_data/{os.path.basename(output_path)}", "w", encoding='utf-8') as file:
        for i, row in df.iterrows():
            file.write(f"Chunk {i+1}:\n")
            for column, value in row.items():
                file.write(f"{column}: {value}\n")
            file.write("\n----------------------------------------------------------------------------------------\n")
    
    print(f"Processed Excel: {file_path}. Chunks saved to {output_path}")

processors = {
    'docx': process_word,
    'doc': process_word,
    # 'pdf': process_pdf,
    'xlsx': process_excel,
    'xls': process_excel,
}

def process_file(file_path: str, processors: Dict[str, Callable[[str], None]]=processors) -> None:
    # absolute_path = os.path.abspath(file_path)
    original_path = Path(file_path)
    extension = original_path.suffix.lower()[1:]
    
    processor = processors.get(extension)
    if processor:
        processor(str(original_path))
    else:
        print(f"No processor found for extension: {extension}")

def process_dir(dir_path: str, processors: Dict[str, Callable[[str], None]], num_workers: int = None) -> None:
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        print(f"The provided path is not a directory: {dir_path}")
        return

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Processing directory: {dir_path}")
    print(f"Using {num_workers} workers")

    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    print(f"Found {len(file_list)} files")

    process_func = partial(process_file, processors=processors)
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_func, file_list)

    print("Finished processing all files")

if __name__ == '__main__':
    # Define your processors dictionary here
    processors = {
    'docx': process_word,
    'doc': process_word,  # python-docx can handle .doc files as well
    # 'pdf': process_pdf,
        'xlsx': process_excel,
        'xls': process_excel,
    }

    # Call the function
    process_dir("data", processors)