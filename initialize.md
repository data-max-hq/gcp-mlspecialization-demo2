# Initialize and Run the Vertex AI TFX Pipeline

This guide will help you initialize and run the Vertex AI TFX pipeline for the Demo 2. Please ensure you have set up your GCP account with the necessary permissions before proceeding.

## Steps to Initialize and Run the Pipeline

1.  Run `startup.sh`

The startup.sh script performs the following actions:

  - Installs pyenv to manage Python versions.
  - Sets the required Python version for the project.
  - Creates a virtual environment (venv).
  - Installs all required Python packages specified in the requirements.txt file.

```
bash startup.sh
```

2. Define the Pipeline
Run the pipeline_definition.py script to define the pipeline components and structure. This script sets up the necessary TFX components such as data ingestion, data transformation, model training, evaluation, and deployment.

```
python black_friday_pipeline/pipeline/pipeline_definition.py
```

3. Submit and Run the Pipeline on Vertex AI
After defining the pipeline, submit it to Vertex AI for execution using the pipeline_submit.py script. This script uploads the pipeline components to Google Cloud Storage and triggers the execution on Vertex AI.
```
python black_friday_pipeline/pipeline/pipeline_submit.py
```


## Monitoring and Logs
You can monitor the progress and logs of your pipeline run through the Vertex AI console on GCP. Navigate to the Vertex AI section and select Pipelines to see the status and details of your pipeline runs.
