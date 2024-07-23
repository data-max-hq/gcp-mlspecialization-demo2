from tfx.components import Transform
import tensorflow as tf
import tensorflow_transform as tft
from tfx.proto import transform_pb2


_FEATURE_KEYS = ["Age","City_Category","Gender","Marital_Status","Occupation","Product_Category_1",'Product_Category_2','Product_Category_3',"Stay_In_Current_City_Years"]
# _CATEGORICAL_NUMERICAL_FEATURES = ["Marital_Status","Occupation","Product_Category_1"]
_CATEGORICAL_NUMERICAL_FEATURES = ["Marital_Status","Occupation","Product_Category_1", "Product_Category_2", "Product_Category_3"]
_CATEGORICAL_STRING_FEATURES = ["City_Category","Age","Stay_In_Current_City_Years", "Gender"]
_LABEL_KEY = 'Purchase'
_VOCAB_SIZE = 1000
_OOV_SIZE = 10


def t_name(key):
  """
  Rename the feature keys so that they don't clash with the raw keys when
  running the Evaluator component.
  Args:
    key: The original feature key
  Returns:
    key with '_xf' appended
  """
  return key + '_xf'


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else -1
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def _make_one_hot(x, key):
  """Make a one-hot tensor to encode categorical features.
  Args:
    X: A dense tensor
    key: A string key for the feature in the input
  Returns:
    A dense one-hot tensor as a float list
  """
  integerized = tft.compute_and_apply_vocabulary(x,
          top_k=_VOCAB_SIZE,
          num_oov_buckets=_OOV_SIZE,
          vocab_filename=key, name=key)
  depth = (
      tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE)
  one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(depth, tf.int32),
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded, [-1, depth])



def preprocessing_fn(inputs):

    outputs = {}
       
    for key in _CATEGORICAL_NUMERICAL_FEATURES:
       outputs[t_name(key)] = _make_one_hot(tf.strings.strip(
        tf.strings.as_string(_fill_in_missing(inputs[key]))), key)
       
    for key in _CATEGORICAL_STRING_FEATURES:
       outputs[t_name(key)] = _make_one_hot(_fill_in_missing(inputs[key]), key)

    outputs[_LABEL_KEY] = tft.scale_to_z_score(inputs[_LABEL_KEY])
    batch_size = tf.shape(input=inputs[_LABEL_KEY])[0]

    def feature_from_scalar(value):
      return tf.tile(tf.expand_dims(value, 0), multiples=[batch_size])

    label_mean = tft.mean(inputs[_LABEL_KEY])
    label_var = tft.var(inputs[_LABEL_KEY])

    outputs['label_mean'] = feature_from_scalar(label_mean)
    outputs['label_var'] = feature_from_scalar(label_var)

    return outputs


def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file="components/data_transformation.py",
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'],
            transform=['train', 'eval']
        )
    )
