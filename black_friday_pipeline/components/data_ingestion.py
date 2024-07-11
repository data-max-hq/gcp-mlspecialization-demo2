from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2
from tfx.v1.proto import Output
from tfx.v1.proto import SplitConfig 

def create_example_gen(input_base: str):
    output = Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                SplitConfig.Split(name='train', hash_buckets=8),
                SplitConfig.Split(name='eval', hash_buckets=1),
                SplitConfig.Split(name='test', hash_buckets=1)
             ]))
    
    example_gen = CsvExampleGen(input_base=input_base, output_config=output)
    
    return example_gen