import os

import dotenv
import tensorflow as tf
from absl import logging
from tensorflow_transform import TFTransformOutput
from tfx.components import Trainer
from tfx.proto import trainer_pb2

dotenv.load_dotenv()

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

_LABEL_KEY = "Purchase"
_FEATURE_KEYS = [
    "Age",
    "City_Category",
    "Gender",
    "Marital_Status",
    "Occupation",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
    "Stay_In_Current_City_Years",
]

AGE_GROUP_INDICES = {
    "0-17": 0,
    "18-25": 1,
    "26-35": 2,
    "36-45": 3,
    "46-50": 4,
    "51-55": 5,
    "55+": 6,
}

AGE_GROUP_WEIGHTS = {
    0: 2.0,  # Weight for '0-17'
    6: 1.5,  # Weight for '55+'
    # Default weight is 1.0 for other groups
}


def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""

    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()

        # Remove label feature and other features that will not be present at serving time.
        raw_feature_spec.pop(_LABEL_KEY)
        required_feature_spec = {
            k: v for k, v in raw_feature_spec.items() if k in _FEATURE_KEYS
        }

        raw_features = tf.io.parse_example(serialized_tf_example, required_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info("serve_transformed_features = %s", transformed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        print("Raw feature spec:", raw_feature_spec)

        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info("eval_transformed_features = %s", transformed_features)
        return transformed_features

    return transform_features_fn


def input_fn(file_pattern, tf_transform_output, batch_size=200):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    print("Transformed feature spec:", transformed_feature_spec)

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(
            filenames, compression_type="GZIP"
        ),
        label_key="Purchase",
    )
    print("Dataset element spec:", dataset.element_spec)

    def add_sample_weights(features, label):
        # Extract the 'Age_xf' one-hot encoded feature
        age_one_hot = features["Age_xf"]

        # Determine the index of the active age category in the one-hot vector
        age_index = tf.argmax(age_one_hot, axis=1, output_type=tf.int32)

        # Apply weights based on conditions
        sample_weight = tf.where(
            tf.equal(age_index, AGE_GROUP_INDICES["0-17"]),
            tf.constant(AGE_GROUP_WEIGHTS[0], dtype=tf.float32),
            tf.where(
                tf.equal(age_index, AGE_GROUP_INDICES["55+"]),
                tf.constant(AGE_GROUP_WEIGHTS[6], dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32),  # Default weight for other groups
            ),
        )

        return features, label, sample_weight

    dataset = dataset.map(add_sample_weights)

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
        "serving_default": _get_tf_examples_serving_signature(
            model, tf_transform_output
        ),
        "transform_features": _get_transform_features_signature(
            model, tf_transform_output
        ),
    }

    model.save(output_dir, save_format="tf", signatures=signatures)


def _build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
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
                shape=[None], name=key, dtype=spec.dtype, sparse=True
            )
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(
                shape=spec.shape or [1], name=key, dtype=spec.dtype
            )
        else:
            raise ValueError("Spec type is not supported: ", key, spec)

    output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    output = tf.keras.layers.Dense(100, activation="relu")(output)
    output = tf.keras.layers.Dense(70, activation="relu")(output)
    output = tf.keras.layers.Dense(50, activation="relu")(output)
    output = tf.keras.layers.Dense(20, activation="relu")(output)
    output = tf.keras.layers.Dense(1)(output)

    return tf.keras.Model(inputs=inputs, outputs=output)


def run_fn(fn_args):
    """Train the model based on given args.

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model(tf_transform_output)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",  # Using MSE for regression
        metrics=["mean_absolute_error"],  # MAE as an additional metric
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )
    print("Training logs saved to: " + fn_args.model_run_dir)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping],
    )
    # Ensure the transformation layer is saved with the model
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)


def create_trainer(transform, schema_gen, module_file):
    return Trainer(
        module_file=module_file,
        # Adjust this path
        custom_config={
            "ai_platform_training_args": {
                "project": GOOGLE_CLOUD_PROJECT,
                "region": GOOGLE_CLOUD_REGION,
                "job-dir": f"{GCS_BUCKET_NAME}/jobs",
            }
        },
        transformed_examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=2000),
    )
