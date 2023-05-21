import pandas as pd
from pathlib import Path
import argparse
import os

def preprocess_data():
    parser = argparse.ArgumentParser("preprocessing")
    parser.add_argument("--real_data", type=str, help="Path to fetched data")
    parser.add_argument("--transformed_data", type=str, help="Path of output data")
    args = parser.parse_args()

    ##### Read data
    filename = os.listdir(args.real_data)
    print("real_data_arg: " + str(args.real_data))
    dataset = pd.read_csv((Path(args.real_data) / filename[0]))

    ##### Remove duplicates
    dataset = dataset[dataset.duplicated() == False]

    ##### Lower case letters
    dataset.iloc[:, 0] = dataset.iloc[:, 0].str.lower()

    #### Get the first 100 records
    dataset = dataset.head(100)
    print(len(dataset))
    # dataset.to_csv(f"{Path(__file__).parent}/cairs_processed.csv", index = False)

    transformed_data = dataset.to_csv(
        (Path(args.transformed_data) / "transformed_data.csv"), index = False
    )