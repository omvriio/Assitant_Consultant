import os
import pandas as pd 
file_path='exhi.xlsx'
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
df = pd.read_excel(current_dir+"\\"+ file_path,nrows=2)
print(df)