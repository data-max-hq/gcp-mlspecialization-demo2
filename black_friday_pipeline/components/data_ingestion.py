from tfx.components import CsvExampleGen
import tfx.proto

def create_example_gen(input_base: str):
    output = tfx.proto.Output(
             split_config=tfx.proto.example_gen_pb2.SplitConfig(splits=[
                 tfx.proto.SplitConfig.Split(name='train', hash_buckets=8),
                 tfx.proto.SplitConfig.Split(name='eval', hash_buckets=1),
                 tfx.proto.SplitConfig.Split(name='test', hash_buckets=1)
             ]))
    
    example_gen = CsvExampleGen(input_base=input_base, output_config=output)
    
    return example_gen