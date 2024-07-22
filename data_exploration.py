import tensorflow_data_validation as tfdv
from tfx.components.statistics_gen import executor as statistics_gen_executor

# Load and display statistics

statistics_uri = "gs://dataset_bucket_demo2/pipeline_root/black_friday_pipeline/908149789490/black-friday-pipeline-20240722092631/StatisticsGen_6603971401143549952/statistics"
schema_uri = "gs://dataset_bucket_demo2/pipeline_root/black_friday_pipeline/908149789490/black-friday-pipeline-20240722092631/SchemaGen_-7231086654138613760/schema"
anomalies_uri = "gs://dataset_bucket_demo2/pipeline_root/black_friday_pipeline/908149789490/black-friday-pipeline-20240722092631/ExampleValidator_-1466479131104378880/anomalies/Split-train"




# Check if the default slice is present in the statistics
stats = tfdv.load_statistics(statistics_uri)
if not any(slice_stats.slice == statistics_gen_executor.DEFAULT_SLICE_NAME for slice_stats in stats.datasets):
    raise ValueError('Missing default slice')



schema = tfdv.load_schema_text(schema_uri)
tfdv.display_schema(schema)

anomalies = tfdv.load_anomalies_text(anomalies_uri)
tfdv.display_anomalies(anomalies)