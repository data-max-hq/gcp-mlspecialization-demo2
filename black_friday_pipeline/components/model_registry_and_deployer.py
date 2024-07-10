# File: components/model_registry_and_deployer.py
from tfx.dsl.component.experimental.decorators import component
from tfx.types.standard_artifacts import PushedModel
from google.cloud import aiplatform
from typing import TypedDict

class VertexAIRegisterDeployOutputs(TypedDict):
    test: str

@component
def create_register_and_deployer() -> VertexAIRegisterDeployOutputs:
    

    print("Registering and deploying the model to Vertex AI. CHECK TEST")
    pushed_model = "gs://dataset_bucket_demo2/serving_model/black_friday_pipeline/1720540282"
    project = 'indigo-idea-428211-h3'
    region = 'europe-west3'

    aiplatform.init(project=project, location=region)

    # Register the model
    model_upload = aiplatform.Model.upload(
        display_name="black_friday_model",
        artifact_uri=pushed_model,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest',
    )

    # Find or create the endpoint
    endpoints = aiplatform.Endpoint.list(filter='display_name="black_friday_endpoint"')
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name="black_friday_endpoint")

    # Deploy the model to the endpoint
    model_upload.deploy(endpoint=endpoint)

    return {
        'test': 'Success'
    }
