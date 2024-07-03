from pipeline.pipeline_definition import create_pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

pipeline_name = 'black_friday_pipeline'
pipeline_root = 'gs://your-bucket/tfx_pipeline_output'
data_path = 'gs://your-bucket/black_friday_dataset'
metadata_path = 'gs://your-bucket/tfx_metadata/metadata.db'

pipeline = create_pipeline(pipeline_name, pipeline_root, data_path, metadata_path)

LocalDagRunner().run(pipeline)
