import pandas as pd
from pathlib import Path
import argparse
import os
def preprocess_data(real_data_path, transformed_data_path):
    ##### Read data
    filename = os.listdir(real_data_path)
    dataset = pd.read_csv((Path(real_data_path) / filename[0]))

    ##### Remove duplicates
    dataset = dataset[dataset.duplicated() == False]

    ##### Lower case letters
    dataset.iloc[:, 0] = dataset.iloc[:, 0].str.lower()

    #### Get the first 100 records
    dataset = dataset.head(100)

    transformed_data = dataset.to_csv(
        (Path(transformed_data_path) / "transformed_data.csv"), index = False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocessing")
    parser.add_argument("--real_data", type=str, help="Path to fetched data")
    parser.add_argument("--transformed_data", type=str, help="Path of output data")
    args = parser.parse_args()

    preprocess_data(args.real_data, args.transformed_data)