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
"""Tests for data_pipeline_utils."""

import tempfile

import data_pipeline_utils as utils
import mock
import numpy as np
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from google.protobuf import text_format

_TEST_SCHEMA = b"""
  feature {
    name: "id_col"
    value_count {
      min: 1
      max: 1
    }
    type: INT
    presence {
      min_fraction: 1.0
      min_count: 1
    }
  }
  feature {
    name: "cat_col1"
    value_count {
      min: 1
      max: 1
    }
    type: BYTES
    presence {
      min_fraction: 1.0
      min_count: 1
    }
  }
  feature {
    name: "num_col1"
    value_count {
      min: 1
      max: 1
    }
    type: FLOAT
    presence {
      min_count: 1
    }
  }
"""


class DataPipelineUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(DataPipelineUtilsTest, self).setUp()
    self._schema = schema_pb2.Schema()
    text_format.Parse(_TEST_SCHEMA, self._schema)

  def test_make_transformed_key(self):
    input_key = 'key'
    expected_key = 'tr_key'
    self.assertEqual(utils.make_transformed_key(input_key), expected_key)

  def test_get_transformed_keys(self):
    input_keys = ['key1', 'key2']
    expected_keys = ['tr_key1', 'tr_key2']
    self.assertListEqual(utils.get_transformed_keys(input_keys), expected_keys)

  def test_get_raw_feature_spec_train_mode(self):
    expected = {
        u'cat_col1': tf.VarLenFeature(dtype=tf.string),
        u'id_col': tf.VarLenFeature(dtype=tf.int64),
        u'num_col1': tf.VarLenFeature(dtype=tf.float32)
    }
    actual = utils.get_raw_feature_spec(self._schema,
                                        tf.estimator.ModeKeys.TRAIN)
    self.assertDictEqual(actual, expected)

  @mock.patch('data_pipeline_utils.features_config')
  def test_get_raw_feature_spec_predict_mode(self, feature_config):
    feature_config.TARGET_FEATURE = 'num_col1'
    expected = {
        u'cat_col1': tf.VarLenFeature(dtype=tf.string),
        u'id_col': tf.VarLenFeature(dtype=tf.int64)
    }
    actual = utils.get_raw_feature_spec(self._schema,
                                        tf.estimator.ModeKeys.PREDICT)

    self.assertDictEqual(actual, expected)

  def test_make_dataset_schema(self):
    generated_dataset_schema = utils.make_dataset_schema(
        self._schema, tf.estimator.ModeKeys.TRAIN)

    self.assertIsInstance(generated_dataset_schema, dataset_schema.Schema)

  def test_read_schema(self):
    temp_schema_file = tempfile.NamedTemporaryFile(
        dir=tempfile.mkdtemp(), delete=False)
    temp_schema_file.write(_TEST_SCHEMA)
    temp_schema_file.close()
    expected_schema = schema_pb2.Schema()
    text_format.Parse(_TEST_SCHEMA, expected_schema)
    actual_schema = utils.read_schema(temp_schema_file.name)

    self.assertEqual(actual_schema, expected_schema)

  @mock.patch('data_pipeline_utils.features_config')
  def test_make_csv_coder_train_mode(self, feature_config):
    feature_config.TARGET_FEATURE = 'num_col1'
    feature_config.ALL_FEATURES = ['id_col', 'cat_col1', 'num_col1']
    # Assert that generated csv_coder is instance of tft_coders.CsvCoder.
    csv_coder = utils.make_csv_coder(self._schema, tf.estimator.ModeKeys.TRAIN)

    self.assertIsInstance(csv_coder, tft_coders.CsvCoder)

    # Assert that csv_coder contains all feature columns.
    expected_columns = feature_config.ALL_FEATURES
    self.assertListEqual(csv_coder._column_names, expected_columns)

  @mock.patch('data_pipeline_utils.features_config')
  def test_make_csv_coder_predict_mode(self, feature_config):
    feature_config.TARGET_FEATURE = 'num_col1'
    feature_config.ALL_FEATURES = ['id_col', 'cat_col1', 'num_col1']
    expected_columns = ['id_col', 'cat_col1']
    csv_coder = utils.make_csv_coder(self._schema,
                                     tf.estimator.ModeKeys.PREDICT)

    # Assert that target column is removed from csv_coder column_names.
    self.assertListEqual(csv_coder._column_names, expected_columns)

  def test_replace_missing_values(self):
    a = tf.constant([1.0], dtype=tf.float32)
    b = tf.constant(['Test'], dtype=tf.string)
    indices = [[0, 0]]
    shape = [1, 1]
    input_tensors = [
        tf.SparseTensor(indices=indices, values=a, dense_shape=shape),
        tf.SparseTensor(indices=indices, values=b, dense_shape=shape)
    ]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      actual = [
          sess.run(utils.replace_missing_values(tensor))
          for tensor in input_tensors
      ]
      expected = [
          np.array([1.0], dtype=np.float32),
          np.array([b'Test'], dtype=np.object)
      ]
      self.assertListEqual(actual, expected)


if __name__ == '__main__':
  tf.test.main()
