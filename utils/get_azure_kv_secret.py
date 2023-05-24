from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
import configparser

def get_kv_secret(credential, secret_name):
    vault_url = "https://kv-05559-s-adf.vault.azure.net"
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    access_token = secret_client.get_secret(secret_name).value
    return access_token

def get_azure_secret_value(secret_name):
    config = configparser.ConfigParser()
    config.read('../azure.properties')

    # Access the connection values
    tenant_id = config.get('ServicePrincipal', 'tenant_id')
    client_id = config.get('ServicePrincipal', 'client_id')
    client_secret = config.get('ServicePrincipal', 'client_secret')
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    # HuggingFace access token
    access_token = get_kv_secret(credential, secret_name)
    return access_token