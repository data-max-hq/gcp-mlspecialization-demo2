from typing import Any, Dict, TypedDict
from tfx.v1.dsl.components import component
from tfx.types.standard_artifacts import PushedModel
from tfx.v1.dsl.components import InputArtifact, OutputArtifact
from google.cloud import aiplatform

class VertexAIRegisterDeployOutputs(TypedDict):
    pass


@component
def create_register_and_deployer(
    pushed_model: InputArtifact[PushedModel]
) -> VertexAIRegisterDeployOutputs:
    
    
    project = 'indigo-idea-428211-h3'
    region = 'europe-west3'

    aiplatform.init(project=project, location=region)

    # Register the model
    model_upload = aiplatform.Model.upload(
        display_name="black_friday_model",
        artifact_uri=pushed_model.uri,
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

    return {}
