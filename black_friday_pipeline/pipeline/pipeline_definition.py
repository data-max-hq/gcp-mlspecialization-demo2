from components.data_ingestion import create_example_gen
from components.data_transformation import create_transform
from components.data_validation import create_data_validation
from components.model_evaluator_and_pusher import create_evaluator_and_pusher
from components.model_trainer import create_trainer
from tfx.orchestration import pipeline


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    serving_model_dir: str,
    module_file: str,
    project: str,
    region: str,
):
    example_gen = create_example_gen(data_path)
    statistics_gen, schema_gen, example_validator = create_data_validation(example_gen)
    transform = create_transform(example_gen, schema_gen)
    trainer = create_trainer(transform, schema_gen, module_file)
    evaluator, pusher, resolver = create_evaluator_and_pusher(
        example_gen, trainer, serving_model_dir
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            resolver,
            evaluator,
            pusher,
        ],
    )
