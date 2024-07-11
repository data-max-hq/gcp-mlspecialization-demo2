from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2 
import tensorflow_model_analysis as tfma


eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='Purchase')
    ],
    slicing_specs=[
        tfma.SlicingSpec()
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='RootMeanSquaredError', threshold=tfma.MetricThreshold(value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 1})))
                ])]
        )


def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config,
        splits_config=evaluator_pb2.SplitsConfig(
            evaluate=['test']
        )
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )
    return evaluator, pusher
