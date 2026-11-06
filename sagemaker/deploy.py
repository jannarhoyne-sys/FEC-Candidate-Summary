import json
import time
from pathlib import Path

import boto3

# Settings
REGION = "us-east-2"
ACCOUNT = "576489586222"
IMAGE_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/house-vote-share:latest"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/SageMakerExecutionRole-house"

ENDPOINT_NAME = "house-vote-share-endpoint"
STAMP = time.strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"house-vote-share-model-{STAMP}"
CONFIG_NAME = f"house-vote-share-config-{STAMP}"

MEMORY_MB = 3072
MAX_CONCURRENCY = 1

# sample race to predict (the same format the Flask app's /invocations route accepts)
SAMPLE = {
    "state": "CA",
    "candidates": [
        {"name": "Candidate A", "party": "DEM", "status": "INCUMBENT", "raised": 2000000},
        {"name": "Candidate B", "party": "REP", "status": "CHALLENGER", "raised": 500000},
    ],
}

sm = boto3.client("sagemaker", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def endpoint_status(name):
#Return the endpoint's status, or None if it doesn't exist
    try:
        return sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
    except sm.exceptions.ClientError:
        return None


status = endpoint_status(ENDPOINT_NAME)

if status is None:
    print(f"1. Creating model {MODEL_NAME}")
    print(f"   image: {IMAGE_URI}")
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={"Image": IMAGE_URI},
        ExecutionRoleArn=ROLE_ARN,
    )

    print(f"2. Creating serverless endpoint config {CONFIG_NAME}")
    sm.create_endpoint_config(
        EndpointConfigName=CONFIG_NAME,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "ServerlessConfig": {"MemorySizeInMB": MEMORY_MB, "MaxConcurrency": MAX_CONCURRENCY},
        }],
    )

    print(f"3. Creating endpoint {ENDPOINT_NAME} (this takes a few minutes)...")
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=CONFIG_NAME)
else:
    print(f"Endpoint {ENDPOINT_NAME} already exists (status: {status}); reusing it.")

while True:
    status = endpoint_status(ENDPOINT_NAME)
    print(f"   status: {status}")
    if status == "InService":
        break
    if status == "Failed":
        reason = sm.describe_endpoint(EndpointName=ENDPOINT_NAME).get("FailureReason")
        raise SystemExit(f"Endpoint failed: {reason}\n"
                         f"Check CloudWatch logs: /aws/sagemaker/Endpoints/{ENDPOINT_NAME}")
    time.sleep(30)

# sample prediction
print("\n4. Sending a sample race to the endpoint:")
print(json.dumps(SAMPLE, indent=2))

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(SAMPLE),
)
result = json.loads(response["Body"].read())

print("\nPrediction from SageMaker:")
print(json.dumps(result, indent=2))

# save input and output
out = Path(__file__).parent / "sample_prediction.json"
out.write_text(json.dumps({"endpoint": ENDPOINT_NAME, "image": IMAGE_URI,
                           "input": SAMPLE, "output": result}, indent=2))
print(f"\nSaved to {out}")
print("When you're done, run:  python sagemaker/cleanup.py")
