from pathlib import Path
import pandas as pd
import argparse
def data_preparations():
    parser = argparse.ArgumentParser("read_data")
    parser.add_argument("--fetched_data", type=str, help="Path of fetched data")
    args = parser.parse_args()

    data = "https://teststoragelogicapp0123.blob.core.windows.net/test/valid/full_cair_list_with_text_2023_2_16.csv"
    dataset1 = pd.read_csv(data)
    dataset1.to_csv((Path(args.fetched_data) / "fetched_data.csv"), index=False)

