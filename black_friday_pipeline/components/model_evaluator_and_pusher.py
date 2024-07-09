from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2 
import tensorflow_model_analysis as tfma


eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='Purchase')
    ],
    slicing_specs=[
        tfma.SlicingSpec(feature_keys=['Gender'])
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='MeanSquaredError'),
                tfma.MetricConfig(class_name='MeanAbsoluteError'),
                tfma.MetricConfig(class_name='RootMeanSquaredError'),
                tfma.MetricConfig(class_name='MeanAbsolutePercentageError'),
            ],
             thresholds={
                'RootMeanSquaredError': tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        upper_bound={'value': float('inf')}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.LOWER_IS_BETTER,
                        absolute={'value': -float('inf')}
                    )
                )
            }
        )
    ]
)


def create_evaluator_and_pusher(transform, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config
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
