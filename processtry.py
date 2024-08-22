# from unstructured.partition.auto import partition
from unstructured.chunking import Chunker
from langchain_community.document_loaders import UnstructuredPDFLoader as partition
# object=partition("data\DML_00786_22_02466_DML_Radars-2026_SW2_update_release_2 1.xlsx",chunking_strategy="by_title",mode="elements")

# from setup import process_file, process_dir
# import pypandoc

# process_dir("data")
# object=process_file("data/MAIN_00786_22_02502_CS_Radars-2026_SW2_update_release_2.docx")
# print(object)
#print the contennt of the object
object=partition("data/MAIN_00786_22_02502_CS_Radars-2026_SW2_update_release_2.pdf",chunking_strategy="by_title",mode="elements")
# for doc in object.load():
#     print(doc.page_content)
#     print("\n ----------------------------------------------------------------------------------------\n")
with open("output.txt", "w") as file:
    for doc in object.load():
        file.write(doc.page_content)
        file.write("\n ----------------------------------------------------------------------------------------\n")
    

import magic 