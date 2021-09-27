# Copyright 2021 Google LLC
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
"""Feature transformation utility functions."""

import ast
import configparser

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.tfxio import csv_tfxio

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

_TRANSFORMED_KEY_PREFIX = 'tr'


# TODO(zmtbnv): implement type annotations
def make_transformed_key(key):
  """Creates key for transformed feature.

  Args:
    key: Feature key.

  Returns:
    Newly generated key for transformed feature.
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


def get_raw_feature_spec(schema_file, mode, target_feature=None):
  """Retrieves raw feature spec for given schema.

  Args:
    schema_file: Serialized Schema proto file.
    mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
    target_feature: Target feature.

  Returns:
    feature_spec: A dictionary of raw feature spec.
  """
  feature_spec = schema_utils.schema_as_feature_spec(schema_file).feature_spec
  if mode == tf.estimator.ModeKeys.PREDICT:
    feature_spec.pop(target_feature)
  return feature_spec


def make_dataset_schema(schema_file, mode, target_feature):
  """Creates dataset schema given schema proto file.

  Args:
    schema_file: Serialized Schema proto file.
    mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
    target_feature: Target feature.

  Returns:
    Instance of dataset_schema.Schema object.
  """
  feature_spec = get_raw_feature_spec(schema_file, mode, target_feature)
  return schema_utils.schema_from_feature_spec(feature_spec)


def read_schema(file_path):
  """Reads a schema file from specified location.

  Args:
    file_path: The location of the file holding a serialized Schema proto.

  Returns:
    An instance of dataset_schema.Schema object.
  """
  schema = schema_pb2.Schema()
  with tf.io.gfile.GFile(file_path) as f:
    text_format.Parse(f.read(), schema)
  return schema


def make_csv_coder(schema_file, all_features, mode, target_feature=None):
  """Creates instance of CsvCoder.

  Args:
    schema_file: Serialized Schema proto file.
    all_features: All features.
    mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
    target_feature: Target feature.

  Returns:
    Instance of tft.coders.CsvCoder.
  """
  schema = make_dataset_schema(schema_file, mode, target_feature)
  if mode == tf.estimator.ModeKeys.PREDICT:
    features = list(all_features)
    if target_feature in features:
      features.remove(target_feature)
      return csv_tfxio.BeamRecordCsvTFXIO(
          physical_format='text', column_names=features, schema=schema)
  return csv_tfxio.BeamRecordCsvTFXIO(
      physical_format='text', column_names=all_features, schema=schema)


def preprocess_sparsetensor(sp_tensor):
  """Converts given SparseTensor to Dense tensor and reduces it's dimension.

  First, the SparseTensor is converted to a dense tensor. For example, given:
  SparseTensor(indices=[[0, 0]], values=[1.], dense_shape=[1, 1]) we get dense
  tensor of array([[ 1.]], dtype=float32) after conversion. Then,
  tf.squeeze is used to remove dimension of size 1 from tensor's shape.
  For example, given tensor array([[ 1.]], dtype=float32) of rank 2 (matrix),
  we get tensor array([ 1.], dtype=float32) of rank 1 (vector).

  Args:
    sp_tensor: A `SparseTensor` of rank 2.

  Returns:
    A rank 1 tensor where missing values of have been replaced.
  """
  default_value = '' if sp_tensor.dtype == tf.string else 0
  return tft.sparse_tensor_to_dense_with_shape(
      x=sp_tensor, shape=(None, 1), default_value=default_value)


def parse_features_config(config_file):
  """Parses features configuration file.

  Args:
    config_file: A path to configuration file.

  Returns:
    features_config: A dictionary of features configuration.
  """
  parser = configparser.ConfigParser()
  with tf.io.gfile.GFile(config_file, 'r') as f:
    parser.read_string(f.read())
  raw_config = dict(parser['core'])
  features_config = {k: ast.literal_eval(v) for k, v in raw_config.items()}
  return features_config
