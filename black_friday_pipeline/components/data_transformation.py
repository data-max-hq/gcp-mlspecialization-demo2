from tfx.components import Transform

def preprocessing_fn(inputs):
    import tensorflow_transform as tft
    outputs = inputs.copy()
    outputs['Purchase'] = tft.scale_to_z_score(inputs['Purchase'])
    outputs['Gender'] = tft.compute_and_apply_vocabulary(inputs['Gender'])
    outputs['Age'] = tft.compute_and_apply_vocabulary(inputs['Age'])
    outputs['City_Category'] = tft.compute_and_apply_vocabulary(inputs['City_Category'])
    outputs['Product_Category_2'] = tft.fill_missing_with_mode(inputs['Product_Category_2'])
    outputs['Product_Category_3'] = tft.fill_missing_with_mode(inputs['Product_Category_3'])
    return outputs

def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file="components/data_transformation.py"
    )
