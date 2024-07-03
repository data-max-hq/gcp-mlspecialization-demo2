from tfx.components import CsvExampleGen

def create_example_gen(input_base: str):
    return CsvExampleGen(input_base=input_base)
