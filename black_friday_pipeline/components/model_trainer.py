from tfx.components import Trainer
from tfx.proto import trainer_pb2
import os
import dotenv

dotenv.load_dotenv()


GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def run_fn(fn_args):
    import tensorflow as tf
    from tensorflow_transform import TFTransformOutput

    tf_transform_output = TFTransformOutput(fn_args.transform_output)

    def input_fn(file_pattern, tf_transform_output, batch_size=200):
        transformed_feature_spec = (
            tf_transform_output.transformed_feature_spec().copy()
        )
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_feature_spec,
            reader=tf.data.TFRecordDataset,
            label_key='Purchase'
        )
        return dataset

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 40)

    # def parse_function(features, labels):
    #         # Extract the necessary features
    #         feature_columns = ["Age","City_Category","Gender","Marital_Status","Occupation","Product_Category_1","Stay_In_Current_City_Years"]
    #         inputs = [features[feature] for feature in feature_columns]
    #         # Concatenate inputs into a single tensor
    #         concatenated_inputs = tf.concat(inputs, axis=-1)
    #         return concatenated_inputs, labels

    # # Map the parse function to the datasets
    # train_dataset = train_dataset.map(parse_function)
    # eval_dataset = eval_dataset.map(parse_function)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps, validation_data=eval_dataset, validation_steps=fn_args.eval_steps)
    model.save(fn_args.serving_model_dir)



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
