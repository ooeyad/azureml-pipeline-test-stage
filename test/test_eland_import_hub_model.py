import logging
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock, Mock
from argparse import ArgumentParser, Namespace
import sys
sys.path.append("../")
from elastic.eland_import_hub_model import get_arg_parser, get_es_client, deploy_model_to_elastic, get_azure_secret_value, ClientSecretCredential
from elasticsearch import AuthenticationException, Elasticsearch


class TestElandImportHubModel(unittest.TestCase):

    @patch('elastic.eland_import_hub_model.argparse.ArgumentParser', spec=ArgumentParser)
    @patch('elastic.eland_import_hub_model.get_azure_secret_value', return_value='123')
    def test_get_arg_parser(self, mock_azure_secret,mock_argparser):
        # Call the function
        result = get_arg_parser()

        # Assertions
        self.assertIsInstance(result, ArgumentParser)
        mock_argparser.assert_called_with()

    @patch('elastic.eland_import_hub_model.Elasticsearch', spec=Elasticsearch)
    # @patch('elastic.eland_import_hub_model.get_azure_secret_value', return_value='123')
    def test_get_es_client(self, mock_elasticsearch):
        # Create mock cli_args
        logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        cli_args = Namespace(
            url='http://localhost:9200',
            cloud_id="123",
            es_api_key="123",
            es_username='myuser',
            es_password='mypassword',
            insecure=True,
            ca_certs=None
        )

        # Configure mock Elasticsearch client
        mock_client = MagicMock(spec=Elasticsearch)
        mock_elasticsearch.return_value = mock_client
        mock_info = {'cluster_name': 'test-cluster', 'version': {'number': '7.10.2'}}
        mock_client.info.return_value = mock_info

        # Call the function
        result = get_es_client(cli_args, logger)

        # Assertions
        mock_elasticsearch.assert_called_with(request_timeout=300, verify_certs=True,
                                              ca_certs=None,cloud_id='123', api_key='123')
        mock_client.info.assert_called_once()
        self.assertEqual(result, mock_client)

    @patch('elastic.eland_import_hub_model.Elasticsearch', spec=Elasticsearch)
    def test_get_es_client2(self, mock_elasticsearch):
        # Create mock cli_args
        logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        cli_args = Namespace(
            url='http://localhost:9200',
            cloud_id="123",
            es_api_key=None,
            es_username='myuser',
            es_password='mypassword',
            insecure=True,
            ca_certs=None
        )

        # Configure mock Elasticsearch client
        mock_client = MagicMock(spec=Elasticsearch)
        mock_elasticsearch.return_value = mock_client
        mock_info = {'cluster_name': 'test-cluster', 'version': {'number': '7.10.2'}}
        mock_client.info.return_value = mock_info

        # Call the function
        result = get_es_client(cli_args, logger)

        # Assertions
        mock_elasticsearch.assert_called_with(request_timeout=300, verify_certs=True,
                                              ca_certs=None, cloud_id='123', basic_auth=('myuser', 'mypassword'))
        mock_client.info.assert_called_once()
        self.assertEqual(result, mock_client)

    @patch('elastic.eland_import_hub_model.check_es_model_exists')
    @patch('elastic.eland_import_hub_model.get_es_client')
    @patch('elastic.eland_import_hub_model.get_arg_parser')
    # @patch('elastic.eland_import_hub_model.logger')
    @patch('elastic.eland_import_hub_model.tempfile.TemporaryDirectory')
    @patch('elastic.eland_import_hub_model.TransformerModel')
    @patch('elastic.eland_import_hub_model.PyTorchModel')
    @patch('elastic.eland_import_hub_model.req.request')
    def test_deploy_model_to_elastic(self, mock_request, mock_pytorch_model, mock_transformer_model,
                                     mock_temporary_directory, mock_parser, mock_es_client, mock_es_model_check):
        # Mock the necessary objects and functions
        mock_temp_dir = mock_temporary_directory.return_value.__enter__.return_value
        mock_get_arg_parser = mock.Mock(return_value=mock.Mock(parse_args=mock.Mock(return_value=mock.Mock(
            url='http://example.com',
            cloud_id='cloud-id',
            es_api_key='api-key',
            es_username='es-username',
            es_password='password',
            es_model_id='es-model-id',
            hub_model_id='hub-model-id',
            task_type='task-type',
            quantize=False,
            start=True,
            clear_previous=True,
            insecure=False,
            ca_certs='ca-certs'
        ))))

        # Set up the mocked Elasticsearch client
        mock_es = mock.Mock(spec=Elasticsearch)
        mock_es_client = mock.Mock(return_value=mock_es)
        mock_get_es_client = mock.Mock(return_value=mock_es_client)
        # Set up the mocked PyTorchModel
        mock_ptm = mock_pytorch_model.return_value
        mock_ptm.model_id = 'model-id'
        mock_ptm.elasticsearch_model_id.return_value = 'es-model-id'
        mock_pytorch_model.return_value = mock_ptm

        # Set up the mocked TransformerModel
        model_path = "mock_model.pt"
        config = {"key": "value"}
        vocab_path = "mock_vocab.txt"

        mock_tm = mock_transformer_model.return_value
        mock_tm.elasticsearch_model_id.return_value = 'es-model-id'
        mock_transformer_model.return_value = mock_tm
        mock_transformer_model.return_value.save.return_value = (model_path, config, vocab_path)

        # Set up the mocked request response
        mock_request.return_value.status_code = 200


        mock_parser.parse_args.return_value = None
        mock_es_client.return_value = None
        mock_es_model_check.return_value = 200
        # Execute the function under test
        deploy_model_to_elastic()

        # Assert that the necessary objects and functions were called with the expected arguments
        # mock_logger.info.assert_called_with("Model successfully imported with id 'model-id'")
        mock_tm.save.assert_called_with(mock_temp_dir)

    @patch('elastic.eland_import_hub_model.configparser.ConfigParser')
    @patch('elastic.eland_import_hub_model.get_kv_secret')
    @patch('elastic.eland_import_hub_model.ClientSecretCredential', spec=ClientSecretCredential)

    def test_get_azure_secret_value(self, mock_clientSecretCred, mock_get_kv_secret, mock_config_parser):
        # Mock the return values
        mock_tenant_id = 'mock_tenant_id'
        mock_client_id = 'mock_client_id'
        mock_client_secret = 'mock_client_secret'
        mock_access_token = 'mock_access_token'
        mock_client = MagicMock(spec=ClientSecretCredential)
        mock_clientSecretCred.return_value = mock_client
        mock_config = mock_config_parser.return_value
        mock_config.get.return_value = mock_tenant_id, mock_client_id, mock_client_secret

        mock_get_kv_secret.return_value = mock_access_token

        # Call the function with a sample secret name
        result = get_azure_secret_value('sample_secret')

        # Assert the expected behavior
        self.assertEqual(result, mock_access_token)
        mock_config_parser.assert_called_once_with()
        mock_config.read.assert_called_once_with('../azure.properties')
        self.assertEqual(mock_config.get.call_count, 3)


if __name__ == '__main__':
    unittest.main()
