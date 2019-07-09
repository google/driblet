# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data pipeline utility functions."""

import features_config
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils
from google.protobuf import text_format

_TRANSFORMED_KEY_PREFIX = 'tr'


def make_transformed_key(key):
  """Creates key for transformed features.

  Args:
    key: Feature key.

  Returns:
    New generated key for transformed feature.
  """
  return '{}_{}'.format(_TRANSFORMED_KEY_PREFIX, key)


def get_transformed_keys(keys):
  """Gets keys for transformed features.

  Args:
    keys: Feature keys.

  Returns:
    A list of transformed keys.
  """
  return [make_transformed_key(key) for key in keys]


def get_raw_feature_spec(schema_file, mode):
  """Retrieves raw feature spec for given schema.

  Args:
    schema_file: Serialized Schema proto file.
    mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.

  Returns:
    Dictionary of raw feature spec.
  """
  feature_spec = schema_utils.schema_as_feature_spec(schema_file).feature_spec
  if mode == tf.estimator.ModeKeys.PREDICT:
    feature_spec.pop(features_config.TARGET_FEATURE)
  return feature_spec


def make_dataset_schema(schema_file, mode):
  """Retrieves raw feature spec for given schema.

  Args:
    schema_file: Serialized Schema proto file.
    mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.

  Returns:
    Instance of Schema object.
  """
  feature_spec = get_raw_feature_spec(schema_file, mode)
  return dataset_schema.from_feature_spec(feature_spec)


def read_schema(file_path):
  """Reads a schema file from specified location.

  Args:
    file_path: The location of the file holding a serialized Schema proto.

  Returns:
    An instance of Schema object.
  """
  result = schema_pb2.Schema()
  contents = file_io.read_file_to_string(file_path)
  text_format.Parse(contents, result)
  return result


def make_csv_coder(schema_file, mode):
  """Creates instance of CsvCoder.

  Args:
    schema_file: Serialized Schema proto file.
    mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.

  Returns:
    Instance of CsvCoder.
  """
  schema = make_dataset_schema(schema_file, mode)
  if mode == tf.estimator.ModeKeys.PREDICT:
    features = list(features_config.ALL_FEATURES)
    features.remove(features_config.TARGET_FEATURE)
    return tft_coders.CsvCoder(features, schema)
  return tft_coders.CsvCoder(features_config.ALL_FEATURES, schema)


def preprocess_sparsetensor(tensor):
  """Converts given SparseTensor to Dense tensor and reduces it's dimension.

  First, the SparseTensor is converted to a dense tensor. For example, given:
  SparseTensor(indices=[[0, 0]], values=[1.], dense_shape=[1, 1]) we get dense
  tensor of array([[ 1.]], dtype=float32) after conversion. Then,
  tf.squeeze is used to remove dimension of size 1 from tensor's shape.
  For example, given tensor array([[ 1.]], dtype=float32) of rank 2 (matrix),
  we get tensor array([ 1.], dtype=float32) of rank 1 (vector).

  Args:
    tensor: A `SparseTensor` of rank 2.

  Returns:
    A rank 1 tensor where missing values of have been replaced.
  """
  default_value = '' if tensor.dtype == tf.string else 0
  dense_tensor = tf.sparse.to_dense(tensor, default_value)
  return tf.squeeze(dense_tensor, axis=1)
