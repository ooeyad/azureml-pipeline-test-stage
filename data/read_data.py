from pathlib import Path
import datasets
import pandas as pd
from datasets import Dataset
import os
import argparse
# from mldesigner import command_component, Input, Output

# @command_component(
#     name="data_preparations",
#     version="1",
#     display_name="Data Preparation",
#     description="preparing etc..",
#     environment=dict(
#         conda_file=Path(__file__).parent / "conda.yaml",
#         image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
#     ),
# )

parser = argparse.ArgumentParser("read_data")
parser.add_argument("--fetched_data", type=str, help="Path of fetched data")
args = parser.parse_args()

def data_preparations():
    
    #data =  os.environ['DATA_SRC']
    data = "https://teststoragelogicapp0123.blob.core.windows.net/test/valid/full_cair_list_with_text_2023_2_16.csv"
    
    dataset1 = pd.read_csv(data)

#     dataset1.to_csv("cairs.csv", index=False)
    dataset1.to_csv((Path(args.fetched_data) / "fetched_data.csv"), index = False)

data_preparations()
