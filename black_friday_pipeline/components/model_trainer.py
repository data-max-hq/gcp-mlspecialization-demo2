from tfx.components import Trainer
from tfx.proto import trainer_pb2
import tensorflow as tf
from tensorflow_transform import TFTransformOutput
from absl import logging
import os
import dotenv

dotenv.load_dotenv()


GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

_LABEL_KEY = 'Purchase'
_FEATURE_KEYS = ["Age","City_Category","Gender","Marital_Status","Occupation","Product_Category_1",'Product_Category_2','Product_Category_3',"Stay_In_Current_City_Years"]



def _apply_preprocessing(raw_features, tft_layer):
  transformed_features = tft_layer(raw_features)
  if _LABEL_KEY in raw_features:
    transformed_label = transformed_features.pop(_LABEL_KEY)
    return transformed_features, transformed_label
  else:
    return transformed_features, None


def _get_serve_tf_examples_fn(model, tf_transform_output):
  # We must save the tft_layer to the model to ensure its assets are kept and
  # tracked.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_examples):
    # Expected input is a string which is serialized tf.Example format.
    feature_spec = tf_transform_output.raw_feature_spec()
    # Because input schema includes unnecessary fields like 'species' and
    # 'island', we filter feature_spec to include required keys only.
    required_feature_spec = {
        k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS
    }
    parsed_features = tf.io.parse_example(serialized_tf_examples,
                                          required_feature_spec)

    # Preprocess parsed input with transform operation defined in
    # preprocessing_fn().
    transformed_features, _ = _apply_preprocessing(parsed_features,
                                                   model.tft_layer)
    # Run inference with ML model.
    return model(transformed_features)

  return serve_tf_examples_fn

def input_fn(file_pattern, tf_transform_output, batch_size=200):
        transformed_feature_spec = (
            tf_transform_output.transformed_feature_spec().copy()
        )
        print("Transformed feature spec:", transformed_feature_spec)
        print("File pattern:", file_pattern)

        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_feature_spec,
            reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
            label_key='Purchase'
        )

        print("Dataset element spec:", dataset.element_spec)

        return dataset

def _build_keras_model() -> tf.keras.Model:
    """Creates a CNN Keras model for predicting purchase amount in Black Friday data.

    Returns:
        A Keras Model.
    """
    inputs = [
        tf.keras.layers.Input(shape=(1,), name=key)
        for key in _FEATURE_KEYS
    ]
    
    # Concatenate inputs to create a single input tensor
    concatenated_inputs = tf.layers.concatenate(inputs)
    
    # Reshape the inputs to a suitable shape for CNN
    reshaped_inputs = tf.layers.Reshape((len(_FEATURE_KEYS), 1, 1))(concatenated_inputs)
    
    # Convolutional layers
    x = tf.layers.Conv2D(16, (2, 2), activation='relu')(reshaped_inputs)
    x = tf.layers.Conv2D(32, (2, 2), activation='relu')(x)
    x = tf.layers.Flatten()(x)
    
    # Fully connected layer
    x = tf.layers.Dense(64, activation='relu')(x)
    
    # Output layer for regression
    outputs = tf.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss='mean_squared_error',  # Using MSE for regression
        metrics=['mean_absolute_error']  # MAE as an additional metric
    )
    
    model.summary(print_fn=logging.info)
    return model

def run_fn(fn_args):

    tf_transform_output = TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model()

    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps, validation_data=eval_dataset, validation_steps=fn_args.eval_steps)

    print("Model training completed.Serving model dirL ",fn_args.serving_model_dir)
    signatures = {
      'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)



def create_trainer(transform, schema_gen,module_file):
    return Trainer(
        module_file=module_file, 
        # Adjust this path
        custom_config={
            'ai_platform_training_args': {
                'project': GOOGLE_CLOUD_PROJECT,
                'region': GOOGLE_CLOUD_REGION,
                'job-dir': f'{GCS_BUCKET_NAME}/jobs'
            }
        },
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=200)
    )
