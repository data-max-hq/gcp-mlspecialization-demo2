from typing import Any, Dict, TypedDict
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.base.base_component import BaseComponent
from google.cloud import aiplatform

class VertexAIRegisterDeployOutputs(TypedDict):
    pass


@component
def create_register_and_deployer(
    pusher: BaseComponent,
    project: str,
    region: str,
) -> VertexAIRegisterDeployOutputs:

    aiplatform.init(project=project, location=region)

    # Register the model
    model_upload = aiplatform.Model.upload(
        display_name="black_friday_model",
        artifact_uri=pusher.outputs['pushed_model'].get()[0].uri,
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
