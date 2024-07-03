from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner, KubeflowDagRunnerConfig
from tfx.orchestration import pipeline

from pipeline_definition import create_pipeline

pipeline_name = 'black_friday_pipeline'
pipeline_root = 'gs://dataset_bucket_demo2/tfx_pipeline_output'
data_path = 'gs://dataset_bucket_demo2/dataset/train.csv'
metadata_path = None
gcp_project_id = 'indigo-idea-428211-h3'
gcp_region = 'europe-west3'

runner_config = KubeflowDagRunnerConfig(
    # kubeflow_metadata_config=metadata.sqlite_metadata_connection_config(metadata_path),
    pipeline_root=pipeline_root
)

runner = KubeflowDagRunner(config=runner_config)
pipeline_instance = create_pipeline(pipeline_name, pipeline_root, data_path, metadata_path)
runner.run(pipeline_instance)