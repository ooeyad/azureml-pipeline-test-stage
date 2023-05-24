import unittest
from unittest import mock
from unittest.mock import patch, MagicMock, Mock
from io import StringIO
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import EvalPrediction
from huggingface_hub import login
import sys
sys.path.append("../")
from huggingface.huggingface import perform_training, preprocess_data, compute_metrics
import  huggingface

class MockSecretClass():
    def __call__(self):
        pass
    value = '123'

class MockModel:
    def __call__(self, input_ids, labels):
        pass
    def __init__(self, modelId):
        self.modelId = modelId


class MockArgs:
    def __init__(self, prepped_data, status_output):
        self.prepped_data = prepped_data
        self.status_output = status_output

class TestYourScript(unittest.TestCase):
    @patch('sys.stdout', new_callable=StringIO)
    def test_perform_training(self, mock_stdout):
        # Mock the required dependencies or provide sample inputs
        args = MockArgs(prepped_data='prepped_data', status_output='status_output')
        filename = 'dataset.csv'
        dataset1 = pd.DataFrame({'CONCATENATED_TEXT': ['example text']})

        with patch('huggingface.huggingface.get_args', return_value=args), \
                patch('huggingface.huggingface.os.listdir', return_value=[filename]), \
                patch('huggingface.huggingface.pd.read_csv', return_value=dataset1), \
                patch('huggingface.huggingface.Dataset.from_dict') as mock_from_dict, \
                patch('huggingface.huggingface.pd.DataFrame.to_csv', return_value=None), \
                patch('huggingface.huggingface.AutoTokenizer.from_pretrained') as mock_from_pretrained, \
                patch('huggingface.huggingface.Trainer') as mock_trainer:
            # Create a dummy instance of the Dataset class
            dataset1 = pd.DataFrame({'CONCATENATED_TEXT': ["testing", "testing2", "testing3", "testing4"], 'B': ["0", "1", "1", "0"]})
            train1 = dataset1.sample(n=4)
            train = Dataset.from_dict(train1)

            # Mock the AutoTokenizer.from_pretrained method
            mock_tokenizer = mock_from_pretrained.return_value

            # Mock the decode method of the tokenizer
            mock_decode = mock_tokenizer.decode

            # Set the return value of the mock DatasetDict
            dataset_dict = DatasetDict({"train": train, "test": train, "validation": train})

            # Patch the dataset map method to return the mock DatasetDict
            with patch('huggingface.huggingface.datasets.DatasetDict.map', return_value=dataset_dict), \
                 patch('huggingface.huggingface.datasets.DatasetDict.set_format', return_value=dataset_dict), \
                    patch('huggingface.huggingface.AutoModelForSequenceClassification.from_pretrained', return_value=MockModel(modelId='model1')), \
                    patch.object(huggingface.huggingface.huggingface_hub.HfApi, 'list_models', return_value=[MockModel(modelId='model1'), MockModel(modelId='model2')]), \
                    patch('huggingface.huggingface.SecretClient.get_secret', return_value=MockSecretClass()), \
                    patch('huggingface.huggingface.login', return_value=None):

                perform_training()

            # Assert any other assertions as needed

    @mock.patch("huggingface.huggingface.AutoTokenizer.from_pretrained")
    def test_preprocess_data(self, mock_tokenizer):
        # Mock data
        examples = {
            'CONCATENATED_TEXT': 'example text',
            'label1': 1,
            'label2': 0
        }
        labels = ['label1', 'label2']
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        with patch('huggingface.huggingface.AutoTokenizer.from_pretrained') as mock_from_pretrained:
            tokenizer = mock_from_pretrained.return_value
            # Call the function
            result = preprocess_data(examples,labels,tokenizer)

    def test_compute_metrics(self):
        # Mock data
        predictions = [[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]]
        labels = [[0, 1], [1, 0], [1, 0]]
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)

        # Call the function
        result = compute_metrics(eval_prediction)


if __name__ == '__main__':
    unittest.main()
