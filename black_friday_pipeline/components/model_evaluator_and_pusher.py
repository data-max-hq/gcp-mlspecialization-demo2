from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2
from tfx.proto import evaluator_pb2

def create_evaluator_and_pusher(example_gen, trainer):
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(
            specs=[evaluator_pb2.SingleSlicingSpec(column_for_slicing=['Gender'])]
        )
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='gs://your-bucket/saved_models'
            )
        )
    )
    return evaluator, pusher
