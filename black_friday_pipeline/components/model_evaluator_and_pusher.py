from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2, evaluator_pb2
from tfx.proto import model_eval_lib_pb2 as me_proto


def create_evaluator_and_pusher(example_gen, trainer, serving_model_dir):
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(
            specs=[evaluator_pb2.SingleSlicingSpec(column_for_slicing=['Gender'])]
        ),
        eval_config=me_proto.EvalConfig(
            model_specs=[me_proto.ModelSpec(signature_name='serving_default')]
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
