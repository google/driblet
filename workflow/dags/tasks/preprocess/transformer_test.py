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
"""Tests for transformer."""

import glob
import os
import shutil
import sys
import tempfile
import unittest

import data_pipeline_utils as utils
import mock
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
import transformer

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
_TRANSFORM_TEMP_DIR = tempfile.mkdtemp()
_SAVED_MODEL_DIR = os.path.join(_TRANSFORM_TEMP_DIR, 'transform_fn')


def _create_input_metadata(features_config):
  """Construct a DatasetMetadata from a feature spec.

  Args:
    features_config: Features configuration mock.

  Returns:
    A `tf.tf_metadata.dataset_metadata.DatasetMetadata` object.
  """
  feature_spec = {
      features_config.TARGET_FEATURE: tf.VarLenFeature(dtype=tf.float32),
      features_config.ID_FEATURE: tf.VarLenFeature(dtype=tf.int64),
  }
  feature_spec.update({
      feature: tf.VarLenFeature(dtype=tf.float32)
      for feature in features_config.NUMERIC_FEATURES
  })
  feature_spec.update({
      feature: tf.VarLenFeature(dtype=tf.string)
      for feature in features_config.CATEGORICAL_FEATURES
  })
  schema = dataset_schema.from_feature_spec(feature_spec)
  return dataset_metadata.DatasetMetadata(schema)


def _create_output_metadata(features_config, min_value, max_value):
  """Constructs a custom DatasetMetadata.

  Args:
    features_config: Features configuration mock.
    min_value: Minimum value for IntDomain.
    max_value: Maximum value for IntDomain.

  Returns:
    A `tft.tf_metadata.dataset_metadata.DatasetMetadata` object.
  """
  schema = {
      features_config.TARGET_FEATURE:
          dataset_schema.ColumnSchema(
              tf.float32, [], dataset_schema.FixedColumnRepresentation()),
      features_config.ID_FEATURE:
          dataset_schema.ColumnSchema(tf.int64, [None],
                                      dataset_schema.ListColumnRepresentation())
  }
  schema.update({
      utils.make_transformed_key(feature): dataset_schema.ColumnSchema(
          tf.float32, [], dataset_schema.FixedColumnRepresentation())
      for feature in features_config.NUMERIC_FEATURES
  })
  categorical_col_schema = dataset_schema.ColumnSchema(
      dataset_schema.IntDomain(
          tf.int64, min_value, max_value, is_categorical=True), [],
      dataset_schema.FixedColumnRepresentation())
  schema.update({
      utils.make_transformed_key(feature): categorical_col_schema
      for feature in features_config.CATEGORICAL_FEATURES
  })
  return dataset_metadata.DatasetMetadata(schema)


class TestTransformer(tft_unit.TransformTestCase):

  @classmethod
  def setUpClass(cls):
    """Sets up materialized preprocessed data for unit tests.

    The function is called once before the tests start.

    """
    super(TestTransformer, cls).setUpClass()
    transformer_args = [
        '--runner=DirectRunner', '--project=None', '--job_name=test_job',
        '--all-data=%s' % os.path.join(_TEST_DATA_DIR, 'data_all.csv'),
        '--train-data=%s' % os.path.join(_TEST_DATA_DIR, 'data_train.csv'),
        '--eval-data=%s' % os.path.join(_TEST_DATA_DIR, 'data_eval.csv'),
        '--predict-data=%s' % os.path.join(_TEST_DATA_DIR, 'data_predict.csv'),
        '--data-source=csv',
        '--schema-file=%s' % os.path.join(_TRANSFORM_TEMP_DIR, 'schema.pbtxt'),
        '--transform-dir=%s' % _TRANSFORM_TEMP_DIR,
        '--output-dir=%s' % _TRANSFORM_TEMP_DIR, '--generate-schema'
    ]
    transformer.main(argv=(['transformer.py'] + transformer_args))
    sys.stdout.flush()
    sys.stderr.flush()

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(_TRANSFORM_TEMP_DIR)

  @tft_unit.named_parameters(('NoDeepCopy', False), ('WithDeepCopy', True))
  @mock.patch('transformer.features_config')
  def test_preprocessing_fn(self, with_deep_copy, features_config):
    # Fake features_config global variables to test their transformed values.
    features_config.TARGET_FEATURE = 't'
    features_config.ID_FEATURE = 'i'
    features_config.NUMERIC_FEATURES = ['n1', 'n2']
    features_config.CATEGORICAL_FEATURES = ['c1', 'c2']
    features_config.OOV_SIZE = 5
    features_config.VOCAB_SIZE = 10
    input_metadata = _create_input_metadata(features_config)
    input_data = [{
        't': [0.0],
        'i': [0],
        'n1': [1.0],
        'n2': [2.0],
        'c1': ['test1'],
        'c2': ['test2']
    }, {
        't': [1.0],
        'i': [1],
        'n1': [3.0],
        'n2': [4.0],
        'c1': ['test2'],
        'c2': ['test1']
    }]
    expected_data = [{
        't': 0.0,
        'i': [0],
        'tr_n1': -1.0,
        'tr_n2': -1.0,
        'tr_c1': 1,
        'tr_c2': 0
    }, {
        't': 1.0,
        'i': [1],
        'tr_n1': 1.0,
        'tr_n2': 1.0,
        'tr_c1': 0,
        'tr_c2': 1
    }]
    expected_metadata = _create_output_metadata(features_config, 0, 6)
    # Assert that transformed result matches expected_data & expected_metadata.
    with tft_beam.Context(use_deep_copy_optimization=with_deep_copy):
      self.assertAnalyzeAndTransformResults(
          input_data=input_data,
          input_metadata=input_metadata,
          preprocessing_fn=transformer.preprocessing_fn,
          expected_data=expected_data,
          expected_metadata=expected_metadata)

  def test_transform_train_eval_saved_model(self):
    # Assert that saved model files were created.
    self.assertTrue(os.path.isdir(_SAVED_MODEL_DIR))
    self.assertTrue(
        os.path.isfile(os.path.join(_SAVED_MODEL_DIR, 'saved_model.pb')))

  def test_transform_train_eval_variables(self):
    # Assert that there are no files on transform_fn/variables directory.
    self.assertFalse(os.listdir(os.path.join(_SAVED_MODEL_DIR, 'variables')))

  def test_transform_train_eval_metadata(self):
    # Assert that transformed metadata files were created.
    metadata_dir = os.path.join(_TRANSFORM_TEMP_DIR, 'transformed_metadata')
    self.assertTrue(os.path.isdir(metadata_dir))
    self.assertTrue(
        os.path.isfile(os.path.join(metadata_dir, 'v1-json', 'schema.json')))

  def test_transform_train_eval_tf_record(self):
    # Assert that transformed tfrecord.gz files has been created.
    self.assertTrue(glob.glob(os.path.join(_TRANSFORM_TEMP_DIR, '*.tfrecord')))

  def test_transform_predict_tf_record(self):
    # Assert that transformed files for prediction data have been created.
    self.assertTrue(
        glob.glob(os.path.join(_TRANSFORM_TEMP_DIR, 'predict*.tfrecord')))


if __name__ == '__main__':
  unittest.main()
