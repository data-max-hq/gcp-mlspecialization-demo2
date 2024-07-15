from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import tensorflow as tf
import dotenv
import os


dotenv.load_dotenv()

PROJECT_NAME = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
API_ENDPOINT = os.getenv("API_ENDPOINT")

def serialize_example(input_data):
    feature = {
        'Age': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Age'].encode()])),
        'City_Category': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['City_Category'].encode()])),
        'Gender': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Gender'].encode()])),
        'Marital_Status': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Marital_Status']])),
        'Occupation': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Occupation']])),
        'Product_Category_1': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Product_Category_1']])),
        'Product_Category_2': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Product_Category_2']])),
        'Product_Category_3': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['Product_Category_3']])),
        'Stay_In_Current_City_Years': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Stay_In_Current_City_Years'].encode()])),
        'Product_ID': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data['Product_ID'].encode()])),
        'User_ID': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_data['User_ID']]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = GOOGLE_CLOUD_REGION,
    api_endpoint: str = API_ENDPOINT,
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", prediction)

# Example usage
project = GOOGLE_CLOUD_PROJECT
endpoint_id = ENDPOINT_ID
location = GOOGLE_CLOUD_REGION
api_endpoint = API_ENDPOINT

input_data = {
    'Age': "26",
    'City_Category': 'A',
    'Gender': 'M',
    'Marital_Status': 0,
    'Occupation': 4,
    'Product_Category_1': 3,
    'Product_Category_2': 5,
    'Product_Category_3': 0,
    'Stay_In_Current_City_Years': '2',
    'Product_ID': 'P00248942',
    'User_ID': 1000001
}

serialized_example = serialize_example(input_data)

encoded_example = tf.io.encode_base64(serialized_example).numpy().decode('utf-8')



predict_custom_trained_model_sample(
    project="908149789490",
    endpoint_id=ENDPOINT_ID,
    location=GOOGLE_CLOUD_REGION,
    instances={"examples":{"b64": encoded_example}}
)




