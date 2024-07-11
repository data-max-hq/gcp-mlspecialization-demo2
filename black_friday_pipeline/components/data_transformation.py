from tfx.components import Transform
import tensorflow_transform as tft

_FEATURE_KEYS = ["Age","City_Category","Gender","Marital_Status","Occupation","Product_Category_1",'Product_Category_2','Product_Category_3',"Stay_In_Current_City_Years"]
_LABEL_KEY = 'Purchase'

def preprocessing_fn(inputs):

    outputs = {}

    for key in _FEATURE_KEYS:
        outputs[key] = inputs[key]

    outputs[_LABEL_KEY] = tft.scale_to_z_score(inputs[_LABEL_KEY])


    outputs['Gender'] = tft.compute_and_apply_vocabulary(inputs['Gender'])
    outputs['Age'] = tft.compute_and_apply_vocabulary(inputs['Age'])
    outputs['City_Category'] = tft.compute_and_apply_vocabulary(inputs['City_Category'])
    outputs['Product_Category_2'] = tft.fill_missing(inputs['Product_Category_2'], default_value=-1)
    outputs['Product_Category_3'] = tft.fill_missing(inputs['Product_Category_3'], default_value=-1)
    outputs['Stay_In_Current_City_Years'] = tft.compute_and_apply_vocabulary(inputs['Stay_In_Current_City_Years'])
    outputs['Occupation'] = tft.compute_and_apply_vocabulary(inputs['Occupation'])
    return outputs

def _apply_preprocessing(raw_features, tft_layer):
  transformed_features = tft_layer(raw_features)
  if _LABEL_KEY in raw_features:
    transformed_label = transformed_features.pop(_LABEL_KEY)
    return transformed_features, transformed_label
  else:
    return transformed_features, None


def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file="components/data_transformation.py"
    )
