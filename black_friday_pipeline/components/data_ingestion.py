from tfx.components import CsvExampleGen
from tfx.v1.proto import example_gen_pb2
from tfx.v1.proto import Output
from tfx import proto

def create_example_gen(input_base: str):
    output = Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 proto.SplitConfig.Split(name='train', hash_buckets=8),
                 proto.SplitConfig.Split(name='eval', hash_buckets=1),
                 proto.SplitConfig.Split(name='test', hash_buckets=1)
             ]))
    
    example_gen = CsvExampleGen(input_base=input_base, output_config=output)
    
    return example_gen