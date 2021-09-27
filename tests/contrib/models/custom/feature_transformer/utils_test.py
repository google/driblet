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

"""Tests for driblet.models.custom.feature_transformer.utils."""

import os
import tempfile
import numpy as np
import parameterized
import tensorflow as tf
from tfx_bsl.tfxio import csv_tfxio
from google.protobuf import text_format
from driblet.contrib.models.custom.feature_transformer import utils
from tensorflow_metadata.proto.v0 import schema_pb2

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
  feature {
    name: "num_col2"
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
_TARGET_FEATURE = 'num_col2'
_FEATURE_SPEC = {
    u'cat_col1': tf.io.FixedLenFeature([], dtype=tf.string),
    u'id_col': tf.io.FixedLenFeature([], dtype=tf.int64),
    u'num_col1': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=-1),
    u'num_col2': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=-1)
}


class FeatureTransformerUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
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

  @parameterized.parameterized.expand([
      (tf.estimator.ModeKeys.TRAIN, _FEATURE_SPEC),
      (tf.estimator.ModeKeys.EVAL, _FEATURE_SPEC),
      (tf.estimator.ModeKeys.PREDICT,
       {k: _FEATURE_SPEC[k] for k in _FEATURE_SPEC if k != _TARGET_FEATURE})
  ])
  def test_get_raw_feature_spec(self, mode, expected):
    actual = utils.get_raw_feature_spec(self._schema, mode, _TARGET_FEATURE)

    self.assertDictEqual(actual, expected)

  def test_make_dataset_schema(self):
    """Test if it retrieves raw feature spec for given schema."""
    generated_dataset_schema = utils.make_dataset_schema(
        self._schema, _TARGET_FEATURE, tf.estimator.ModeKeys.TRAIN)

    self.assertIsInstance(generated_dataset_schema, schema_pb2.Schema)

  def test_read_schema(self):
    temp_schema_file = tempfile.NamedTemporaryFile(
        dir=tempfile.mkdtemp(), delete=False)
    temp_schema_file.write(_TEST_SCHEMA)
    temp_schema_file.close()

    expected_schema = schema_pb2.Schema()
    text_format.Parse(_TEST_SCHEMA, expected_schema)
    actual_schema = utils.read_schema(temp_schema_file.name)

    self.assertEqual(actual_schema, expected_schema)

  @parameterized.parameterized.expand([
      (tf.estimator.ModeKeys.TRAIN, list(_FEATURE_SPEC)),
      (tf.estimator.ModeKeys.EVAL, list(_FEATURE_SPEC)),
      (tf.estimator.ModeKeys.PREDICT,
       [k for k in _FEATURE_SPEC if k != _TARGET_FEATURE])
  ])
  def test_make_csv_coder(self, mode, column_names):
    csv_coder = utils.make_csv_coder(self._schema, column_names, mode,
                                     _TARGET_FEATURE)

    # Assert that generated csv_coder is instance of tft.coders.CsvCoder.
    self.assertIsInstance(csv_coder, csv_tfxio.BeamRecordCsvTFXIO)
    # Assert that csv_coder contains the feature columns.
    self.assertListEqual(csv_coder._column_names, column_names)

  def test_preprocess_sparsetensor(self):
    test_value1 = tf.constant([1.0], dtype=tf.float32)
    test_value2 = tf.constant(['Test'], dtype=tf.string)
    indices = [[0, 0]]
    shape = [1, 1]
    input_tensors = [
        tf.SparseTensor(indices=indices, values=test_value1, dense_shape=shape),
        tf.SparseTensor(indices=indices, values=test_value2, dense_shape=shape)
    ]

    with self.test_session() as sess:
      actual = [
          sess.run(utils.preprocess_sparsetensor(sp_tensor))
          for sp_tensor in input_tensors
      ]
      expected = [
          np.array([1.0], dtype=np.float32),
          np.array([b'Test'], dtype=np.object)
      ]

      self.assertListEqual(actual, expected)

  @parameterized.parameterized.expand([([
      'all_features', 'categorical_features', 'forward_features',
      'numeric_features'
  ], list), (['target_feature'], str), (['oov_size', 'vocab_size'], int)])
  def test_parse_features_config(self, keys, expected_type):
    config_file = os.path.join(os.path.dirname(__file__), 'features_config.cfg')
    actual = utils.parse_features_config(config_file)
    for key in keys:
      self.assertIsInstance(actual[key], expected_type)


if __name__ == '__main__':
  tf.test.main()
