from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2

def create_example_gen(input_base: str):
    split_config = example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=2)
    ])
    example_gen = CsvExampleGen(
        input_base=input_base,
        split_config=split_config
    )
    return example_gen(input_base=input_base)