import unittest
from unittest.mock import patch
import argparse
import sys
from pathlib import Path
import os
import pandas as pd
import sys
sys.path.append("../")
from data.read_data import data_preparations

class TestDataPreparations(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("pandas.DataFrame.to_csv")
    def test_data_preparations(self, mock_to_csv, mock_read_csv):
        # Mock the return values of the read_csv function and the to_csv method
        mock_read_csv.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_to_csv.return_value = None

        # Create a temporary directory for the fetched data
        fetched_data_path = "test_path"
        os.makedirs(fetched_data_path, exist_ok=True)
        # Run the function with test arguments
        args = argparse.Namespace(fetched_data=fetched_data_path)

        sys.argv = ["program_name", "--fetched_data", str(fetched_data_path)]
        data_preparations()
        # Check if the to_csv method was called with the correct arguments
        mock_to_csv.assert_called_once_with(Path(fetched_data_path) / "fetched_data.csv", index=False)
        # Check if the read_csv function was called with the correct arguments
        mock_read_csv.assert_called_once_with("https://teststoragelogicapp0123.blob.core.windows.net/test/valid/full_cair_list_with_text_2023_2_16.csv")
