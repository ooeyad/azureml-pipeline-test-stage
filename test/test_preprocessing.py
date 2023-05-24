import unittest
from unittest import mock
import pandas as pd
import os
from pathlib import Path
import argparse
import sys
sys.path.append("../")
from prepare_data.preprocessing import preprocess_data, get_args

class TestPreprocessData(unittest.TestCase):
    test_dir = ""

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    # def tearDown(self):
        # Delete the temporary directory and files
        # os.system("rm -rf " + self.test_dir)

    @mock.patch("argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(real_data=test_dir,
                                                transformed_data=test_dir))
    @mock.patch("os.listdir", return_value=["test_data.csv"])
    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.DataFrame.to_csv")
    def test_preprocess_data(self, mock_csv_writer, mock_csv_reader, mock_listdir, mock_args):
        mock_csv_reader.return_value = pd.DataFrame({"col1": ["A", "B", "B", "C", "D", "E"],"col2": [1, 2, 2, 3, 4, 5]})
        mock_csv_writer.return_value = None

        # Call the function with mocked arguments and file operations
        preprocess_data(real_data_path="test_path", transformed_data_path="test_path")

        # Check that the correct files were read and written
        mock_csv_reader.assert_called_once_with(Path("test_path") / "test_data.csv")
        mock_csv_writer.assert_called_once_with(Path("test_path") / "transformed_data.csv", index=False)

    def test_get_args(self):
        args = get_args()