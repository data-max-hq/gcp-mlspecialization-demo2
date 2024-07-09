from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2

import tensorflow_model_analysis as tfma
from components.data_transformation import preprocessing_fn


eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name='eval' and
        # remove the label_key. Note, if using a TFLite model, then you must set
        # model_type='tf_lite'.
        tfma.ModelSpec(label_key='Purchase')
    ],
    slicing_specs=[
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['gender'])
    ])


def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=preprocessing_fn(example_gen.outputs['examples']),
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
