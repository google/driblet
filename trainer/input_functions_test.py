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
"""Tests for input_functions."""

import os
import shutil
import tempfile

import input_functions
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.export import export
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow_transform.saved import saved_transform_io

_TEST_FEATURE_ID = 'id'
_TEST_TARGET_FEATURE = 'test_target'
_TEST_FEATURE = 'test_feature'
_TEST_FEATURE_ID_VALUE = [1]
_TEST_TARGET_FEATURE_VALUE = [1]
_TEST_FEATURE_VALUE = [1.0]
_TEST_DATA_FILE = 'test_data.tfrecord'
_TEST_METADATA_SCHEMA = """
{
  "feature": [{
    "name": "%s",
    "fixedShape": {
      "axis": []
    },
    "type": "INT",
    "domain": {
      "ints": {}
    },
    "parsingOptions": {
      "tfOptions": {
        "varLenFeature": {}
      }
    }
  },{
    "name": "%s",
    "fixedShape": {
      "axis": []
    },
    "type": "INT",
    "domain": {
      "ints": {}
    },
    "parsingOptions": {
      "tfOptions": {
        "fixedLenFeature": {}
      }
    }
  },
  {
    "name": "%s",
    "fixedShape": {
      "axis": []
    },
    "type": "FLOAT",
    "domain": {
      "floats": {}
    },
    "parsingOptions": {
      "tfOptions": {
        "fixedLenFeature": {}
      }
    }
  }]
}
""" % (_TEST_FEATURE_ID, _TEST_TARGET_FEATURE, _TEST_FEATURE)


def _create_test_data():
  """Creates serialized test data in tf.Example format.

  Returns:
    Serialized tf.Example proto.
  """
  feature = {
      _TEST_FEATURE_ID:
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=_TEST_FEATURE_ID_VALUE)),
      _TEST_TARGET_FEATURE:
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=_TEST_TARGET_FEATURE_VALUE)),
      _TEST_FEATURE:
          tf.train.Feature(
              float_list=tf.train.FloatList(value=_TEST_FEATURE_VALUE))
  }
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()


def _write_test_data_to_disk(testfile):
  """Writes test data in tf.Example format to a temporary directory.

  Args:
    testfile: Path to test file.
  """
  test_data = _create_test_data()
  with tf.python_io.TFRecordWriter(testfile) as writer:
    writer.write(test_data)


def _write_schema_to_disk(tempdir):
  """Writes test data schema to temporary a directory.

  Args:
    tempdir: Path to temporary directory.
  """
  test_transform_dir = os.path.join(tempdir, 'transformed_metadata', 'v1-json')
  test_schema = os.path.join(test_transform_dir, 'schema.json')
  file_io.recursive_create_dir(test_transform_dir)
  file_io.write_string_to_file(test_schema, _TEST_METADATA_SCHEMA)


def _create_and_write_test_saved_model(tempdir):
  """Creates test saved model and writes it to disk.

  This test model is used  by `example_serving_receiver_fn` to apply
  transformation to test data.

  Args:
    tempdir: Path to temporary directory.
  """
  export_path = os.path.join(tempdir, 'transform_fn')
  with tf.Graph().as_default():
    with tf.Session().as_default() as session:
      input_placeholder = tf.placeholder(tf.float32, shape=[1])
      output_value = (input_placeholder - 1.0) / 6.0
      input_dict = {
          _TEST_FEATURE_ID: tf.placeholder(tf.int64, shape=[1]),
          _TEST_FEATURE: input_placeholder
      }
      output_dict = {
          _TEST_FEATURE_ID:
              tf.SparseTensor(indices=[[1]], values=[1], dense_shape=[1]),
          'test_scaled_feature':
              output_value
      }
      saved_transform_io.write_saved_transform_from_session(
          session, input_dict, output_dict, export_path)


class InputFunctionsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(InputFunctionsTest, cls).setUpClass()
    cls.tempdir = tempfile.mkdtemp()

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.tempdir)

  def test_convert_sparse_to_dense(self):
    input_tensor = tf.SparseTensor(indices=[[1]], values=[1], dense_shape=[1])
    actual = input_functions.convert_sparse_to_dense(input_tensor)
    self.assertShapeEqual(np.array([[1]]), actual)
    self.assertEqual(actual.dtype, tf.int32)
    self.assertIsInstance(actual, ops.Tensor)

  def test_create_feature_columns(self):
    categorical_features = ['c1']
    numeric_features = ['n1']
    expected_keys = ['tr_c1', 'tr_n1']
    actual_keys = []
    linear_features, dense_features = input_functions.create_feature_columns(
        categorical_features, numeric_features, 2)
    for linear_feature, dense_feature in zip(linear_features, dense_features):
      actual_keys.append(linear_feature.key)
      actual_keys.append(dense_feature.key)
      self.assertIsInstance(linear_feature,
                            feature_column._IdentityCategoricalColumn)
      self.assertIsInstance(dense_feature, feature_column._NumericColumn)
    self.assertListEqual(actual_keys, expected_keys)

  def test_create_feature_columns_list_length(self):
    categorical_features = ['c1', 'c2']
    numeric_features = ['n1', 'n2', 'n3']
    linear_features, dense_features = input_functions.create_feature_columns(
        categorical_features, numeric_features, 2)
    self.assertEqual(len(dense_features), len(numeric_features))
    self.assertEqual(len(linear_features), len(categorical_features))

  def test_get_input_fn(self):
    testfile = os.path.join(self.tempdir, _TEST_DATA_FILE)
    _write_test_data_to_disk(testfile)
    input_fn = input_functions.get_input_fn(testfile, self.tempdir,
                                            _TEST_TARGET_FEATURE,
                                            _TEST_FEATURE_ID, 1, 1)
    features, target = input_fn()
    with self.session() as session:
      features, target = session.run((features, target))
    self.assertEqual(target, _TEST_TARGET_FEATURE_VALUE)
    self.assertEqual(features[_TEST_FEATURE], _TEST_FEATURE_VALUE)
    self.assertDTypeEqual(features[_TEST_FEATURE], np.float32)
    self.assertDTypeEqual(target, np.int64)

  def test_example_serving_receiver_fn(self):
    _write_schema_to_disk(self.tempdir)
    _create_and_write_test_saved_model(self.tempdir)
    raw_feature_spec = {_TEST_FEATURE: tf.FixedLenFeature([], tf.float32)}
    actual = input_functions.example_serving_receiver_fn(
        self.tempdir, raw_feature_spec, _TEST_TARGET_FEATURE, _TEST_FEATURE_ID)
    expected_feature_keys = [_TEST_FEATURE_ID, 'test_scaled_feature']
    actual_keys = sorted(actual.features.keys())
    self.assertIsInstance(actual, export.ServingInputReceiver)
    self.assertListEqual(actual_keys, expected_feature_keys)


if __name__ == '__main__':
  tf.test.main()
