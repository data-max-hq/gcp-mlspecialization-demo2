from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2
from tfx.proto import model_eval_lib_pb2 as me_proto
import tensorflow_model_analysis as tfma


eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name='eval' and
        # remove the label_key. Note, if using a TFLite model, then you must set
        # model_type='tf_lite'.
        tfma.ModelSpec(label_key='Purchase')
    ])


def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(
            specs=[evaluator_pb2.SingleSlicingSpec(column_for_slicing=['Gender'])]
        ),
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
