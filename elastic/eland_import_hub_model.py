#!/usr/bin/env python
"""
Copies a model from the Hugging Face model hub into an Elasticsearch cluster.
This will create local cached copies that will be traced (necessary) before
uploading to Elasticsearch. This will also check that the task type is supported
as well as the model and tokenizer types. All necessary configuration is
uploaded along with the model.
"""

import argparse
import logging
import os
import sys
import tempfile
import textwrap
import json
from elastic_transport.client_utils import DEFAULT
import elasticsearch
from elasticsearch import AuthenticationException, Elasticsearch
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
import time
import requests as req
try:
    from eland.ml.pytorch import PyTorchModel
    from eland.ml.pytorch.transformers import (
        SUPPORTED_TASK_TYPES,
        TaskTypeError,
        TransformerModel,
    )
except ModuleNotFoundError as e:
    exit(1)

MODEL_HUB_URL = "https://huggingface.co"

proxies = {
    'aiproxy.appl.chrysler.com:9080'
}

def get_arg_parser(logger):
    azure_creds = os.environ.get("AZURE_CREDENTIALS")
    es_cloud_id = get_azure_secret_value("es-cloud-id",azure_creds,logger)
    es_username = get_azure_secret_value("es-username",azure_creds,logger)
    es_password = get_azure_secret_value("es-password",azure_creds,logger)
    es_api_key = get_azure_secret_value("es-api-key",azure_creds,logger)


    parser = argparse.ArgumentParser()
    location_args = parser.add_mutually_exclusive_group(required=True)
    location_args.add_argument(
        "--cloud-id",
        default=es_cloud_id,
        help="Cloud ID as found in the 'Manage Deployment' page of an Elastic Cloud deployment",
    )
    parser.add_argument(
        "--hub-model-id",
        required=True,
        default="bart-large-mnli",
        help="The model ID in the Hugging Face model hub, "
             "e.g. dbmdz/bert-large-cased-finetuned-conll03-english",
    )
    parser.add_argument("--es-model-id",
                        required=False,
                        default=None,
                        help="The model ID to use in Elasticsearch, "
                             "e.g. bert-large-cased-finetuned-conll03-english."
                             "When left unspecified, this will be auto-created from the `hub-id`",
                        )
    parser.add_argument(
        "-u", "--es-username",
        required=False,
        default=es_username,
        help="Username for Elasticsearch"
    )
    parser.add_argument(
        "-p", "--es-password",
        required=False,
        default=es_password,
        help="Password for the Elasticsearch user specified with -u/--username"
    )
    parser.add_argument(
        "--es-api-key",
        required=False,
        default="--es-api-key ApiKey " + es_api_key,
        help="API key for Elasticsearch"
    )
    parser.add_argument(
        "--task-type",
        required=False,
        choices=SUPPORTED_TASK_TYPES,
        help="The task type for the model usage. Will attempt to auto-detect task type for the model if not provided. "
             "Default: auto",
        default="zero_shot_classification"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Quantize the model before uploading. Default: False",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        default=False,
        help="Start the model deployment after uploading. Default: False",
    )
    parser.add_argument(
        "--clear-previous",
        action="store_true",
        default=False,
        help="Should the model previously stored with `es-model-id` be deleted"
    )
    parser.add_argument(
        "--insecure",
        action="store_false",
        default=True,
        help="Do not verify SSL certificates"
    )
    parser.add_argument(
        "--ca-certs",
        required=False,
        default=DEFAULT,
        help="Path to CA bundle"
    )

    return parser


def get_es_client(cli_args, logger):
    try:
        es_args = {
            'request_timeout': 300,
            'verify_certs': cli_args.insecure,
            'ca_certs': cli_args.ca_certs
        }
        azure_creds = os.environ.get("AZURE_CREDENTIALS")

        if cli_args.cloud_id:
            es_args['cloud_id'] = cli_args.cloud_id

        es_api_key = get_azure_secret_value("es-api-key",azure_creds,logger)
        es_args['api_key'] = es_api_key

        es_client = Elasticsearch(**es_args)
        es_info = es_client.info()
        logger.info(f"Connected to cluster named '{es_info['cluster_name']}' (version: {es_info['version']['number']})")
        return es_client
    except AuthenticationException as e:
        logger.error(e)
        exit(1)
def deploy_model_to_elastic():
    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    # Parse arguments
    args = get_arg_parser(logger).parse_args()
    # Connect to ES
    logger.info("Establishing connection to Elasticsearch")
    es = get_es_client(args, logger)
    # Trace and save model, then upload it from temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Loading HuggingFace transformer tokenizer and model '{args.hub_model_id}'")

        try:
            tm = TransformerModel(args.hub_model_id, args.task_type, args.quantize)
            model_path, config, vocab_path = tm.save(tmp_dir)
        except TaskTypeError as err:
            logger.error(
                f"Failed to get model for task type, please provide valid task type via '--task-type' parameter. Caused by {err}")
            exit(1)

        ptm = PyTorchModel(es, args.es_model_id if args.es_model_id else tm.elasticsearch_model_id())
        model_exists = check_es_model_exists(es, ptm) == 200

        if model_exists:
            if args.clear_previous:
                try:
                    logger.info(f"Stopping deployment for model with id '{ptm.model_id}'")
                    es.ml.stop_trained_model_deployment(model_id=ptm.model_id, force=True)
                    logger.info(f"Deleting model with id '{ptm.model_id}'")
                    # ptm.delete()
                    es.ml.delete_trained_model(model_id=ptm.model_id,force=True)
                except elasticsearch.ConflictError as err:
                    logger.warning(
                        f"ConflictError: Cannot stop deployment '{ptm.model_id}' as it is referenced by ingest processors."
                    )
                    logger.warning("Consider removing references from ingest processors.")
                    exit(1)

            else:
                logger.error(f"Trained model with id '{ptm.model_id}' already exists")
                logger.info(
                    "Run the script with the '--clear-previous' flag if you want to overwrite the existing model.")
                exit(1)

        wait_time_seconds = 10
        logger.info(f"Waiting for {wait_time_seconds} seconds before starting model deployment.")
        time.sleep(wait_time_seconds)

        logger.info(f"Creating model with id '{ptm.model_id}'")
        ptm.put_config(config=config)

        logger.info(f"Uploading model definition")
        ptm.put_model(model_path)

        logger.info(f"Uploading model vocabulary")
        ptm.put_vocab(vocab_path)

    # Start the deployed model
    if args.start:
        logger.info(f"Starting model deployment")
        ptm.start()

    logger.info(f"Model successfully imported with id '{ptm.model_id}'")
def get_kv_secret(credential, secret_name,logger):
    vault_url = "https://kv-05559-d-adf.vault.azure.net"
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    logger.info(secret_name)
    logger.info("Secret Client: ")
    logger.info(secret_client)
    access_token = secret_client.get_secret(secret_name).value
    return access_token

def get_azure_secret_value(secret_name, azure_credentials,logger):
    credentials = get_azure_credentials(azure_credentials)
    # Access the connection values
    tenant_id = credentials['tenantId']
    client_id = credentials['clientId']
    client_secret = credentials['clientSecret']
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    # HuggingFace access token
    access_token = get_kv_secret(credential, secret_name,logger)
    return access_token

def check_es_model_exists(es, ptm):
    return es.options(ignore_status=404).ml.get_trained_models(model_id=ptm.model_id).meta.status

def get_azure_credentials(str_azure_credentials):
    return json.loads(str_azure_credentials)
if __name__ == "__main__":
    deploy_model_to_elastic()
