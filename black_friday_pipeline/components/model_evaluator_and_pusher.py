from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2 
import tensorflow_model_analysis as tfma


eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key='Purchase',
            preprocessing_function_names=['transform_features'],
            )
        ],
    slicing_specs=[
        tfma.SlicingSpec()
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='RootMeanSquaredError', threshold=tfma.MetricThreshold(value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 100000000000000})))
                ])]
    )


def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config,
        example_splits=['test']
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
