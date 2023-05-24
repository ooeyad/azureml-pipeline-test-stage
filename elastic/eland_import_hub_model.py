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

from elastic_transport.client_utils import DEFAULT
from elasticsearch import AuthenticationException, Elasticsearch
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

def get_arg_parser():
    parser = argparse.ArgumentParser()
    location_args = parser.add_mutually_exclusive_group(required=True)
    location_args.add_argument(
        "--cloud-id",
        # default=os.environ.get("Elastic-05559-d-002:ZWFzdHVzLmF6dXJlLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQ0ODVjZTM5MzFmMWU0YjYzOTNmM2EyNTZkOTZiYTc1ZSQ3Njc4ZmQyZjA2NWI0YWM1OTRhYjVlMmVjMTMxYjI3Mw=="),
        default="Elastic-05559-s-001:ZWFzdHVzMi5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo0NDMkNDlhODhlNTg5YTBhNGJhZDkzNTA0NTFlYmVhZTg3OTckMDdlYjU1NzhjODdlNGI3MWI5NmIwNjY0ZmY3NWI4ODc=",
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
        default="iyad.alswaiti@external.stellantis.com",
        help="Username for Elasticsearch"
    )
    parser.add_argument(
        "-p", "--es-password",
        required=False,
        default="Jordan123456789@",
        help="Password for the Elasticsearch user specified with -u/--username"
    )
    parser.add_argument(
        "--es-api-key",
        required=False,
        default="--es-api-key ApiKey MEowRjhJY0I2dGg1ZG05ZHloNDU6Qmc5ZnJxVUxTRTZEcVBRNjFZa1d6QQ==",
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

        if cli_args.cloud_id:
            es_args['cloud_id'] = cli_args.cloud_id

        # Authentication
        if cli_args.es_api_key:
            es_args['api_key'] = cli_args.es_api_key
        elif cli_args.es_username:
            if not cli_args.es_password:
                logging.error(f"Password for user {cli_args.es_username} was not specified.")
                exit(1)

            es_args['basic_auth'] = (cli_args.es_username, cli_args.es_password)

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
    args = get_arg_parser().parse_args()
    print("done parsing")

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
                logger.info(f"Stopping deployment for model with id '{ptm.model_id}'")
                ptm.stop()

                logger.info(f"Deleting model with id '{ptm.model_id}'")
                ptm.delete()
            else:
                logger.error(f"Trained model with id '{ptm.model_id}' already exists")
                logger.info(
                    "Run the script with the '--clear-previous' flag if you want to overwrite the existing model.")
                exit(1)

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

def check_es_model_exists(es, ptm):
    return es.options(ignore_status=404).ml.get_trained_models(model_id=ptm.model_id).meta.status

if __name__ == "__main__":
    deploy_model_to_elastic()