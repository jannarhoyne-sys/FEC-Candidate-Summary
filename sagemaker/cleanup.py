import boto3

REGION = "us-east-2"
ENDPOINT_NAME = "house-vote-share-endpoint"
PREFIX = "house-vote-share"

sm = boto3.client("sagemaker", region_name=REGION)

try:
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"Deleted endpoint {ENDPOINT_NAME}")
except sm.exceptions.ClientError:
    print(f"No endpoint named {ENDPOINT_NAME} (already deleted)")

for cfg in sm.list_endpoint_configs(NameContains=PREFIX)["EndpointConfigs"]:
    sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
    print(f"Deleted endpoint config {cfg['EndpointConfigName']}")

for model in sm.list_models(NameContains=PREFIX)["Models"]:
    sm.delete_model(ModelName=model["ModelName"])
    print(f"Deleted model {model['ModelName']}")

print("Cleanup done. Check SageMaker AI -> Inference -> Endpoints: the list should be empty.")
