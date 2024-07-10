from typing import Any, Dict, Optional
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.decorators import OutputDict
from tfx.types.standard_artifacts import Model
from google.cloud import aiplatform

@component
def create_register_and_deployer(
    pusher,
    project: str,
    region: str,
) -> OutputDict:

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
