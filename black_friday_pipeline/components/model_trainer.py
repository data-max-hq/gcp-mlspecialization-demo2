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


def _get_tf_examples_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_inference = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(_LABEL_KEY)
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_inference(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).
    return {'outputs': outputs}

  return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn

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

def export_serving_model(tf_transform_output, model, output_dir):
  """Exports a keras model for serving.
  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    model: A keras model to export for serving.
    output_dir: A directory where the model will be exported to.
  """
  # The layer has to be saved to the model for keras tracking purpases.
  model.tft_layer = tf_transform_output.transform_features_layer()

  signatures = {
      'serving_default':
          _get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
  }

  model.save(output_dir, save_format='tf', signatures=signatures)

def _build_keras_model(tf_transform_output: TFTransformOutput
                       ) -> tf.keras.Model:
    """Creates a CNN Keras model for predicting purchase amount in Black Friday data.

    Returns:
        A Keras Model.
    """

    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)

    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=[None], name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=spec.shape or [1], name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)
          
    x = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Adding Dropout for regularization
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=output)



# def _build_keras_model(tf_transform_output: TFTransformOutput
#                        ) -> tf.keras.Model:
#     """Creates a CNN Keras model for predicting purchase amount in Black Friday data.

#     Returns:
#         A Keras Model.
#     """

#     feature_spec = tf_transform_output.transformed_feature_spec().copy()
#     feature_spec.pop(_LABEL_KEY)

#     inputs = {}
#     for key, spec in feature_spec.items():
#         if isinstance(spec, tf.io.VarLenFeature):
#             inputs[key] = tf.keras.layers.Input(
#                 shape=[None], name=key, dtype=spec.dtype, sparse=True)
#         elif isinstance(spec, tf.io.FixedLenFeature):
#             inputs[key] = tf.keras.layers.Input(
#                 shape=spec.shape or [1], name=key, dtype=spec.dtype)
#         else:
#             raise ValueError('Spec type is not supported: ', key, spec)
        

#     # Concatenate inputs to create a single input tensor
#     output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
#     output = tf.keras.layers.Dense(100, activation='relu')(output)
#     output = tf.keras.layers.Dense(70, activation='relu')(output)
#     output = tf.keras.layers.Dense(50, activation='relu')(output)
#     output = tf.keras.layers.Dense(20, activation='relu')(output)
#     output = tf.keras.layers.Dense(1)(output)
#     return tf.keras.Model(inputs=inputs, outputs=output)

def run_fn(fn_args):
   """Train the model based on given args.

   Args:
       fn_args: Holds args used to train the model as name/value pairs.
   """
   tf_transform_output = TFTransformOutput(fn_args.transform_output)
   print("TF Transform output:", tf_transform_output)

   train_dataset = input_fn(
       fn_args.train_files,
       tf_transform_output)
   eval_dataset = input_fn(
       fn_args.eval_files,
       tf_transform_output)

   model = _build_keras_model(tf_transform_output)

   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9)

   model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',  # Using MSE for regression
        metrics=['mean_absolute_error']  # MAE as an additional metric
    )

   tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
   

   model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])
   # Ensure the transformation layer is saved with the model
   export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)

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


