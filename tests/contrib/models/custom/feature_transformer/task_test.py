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
"""Tests for driblet.models.custom.feature_transformer.task."""
import argparse
import glob
import os
import shutil
import tempfile
from absl import flags

import parameterized
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from driblet.contrib.models.custom.feature_transformer import task
from driblet.contrib.models.custom.feature_transformer import utils
from tensorflow_metadata.proto.v0 import schema_pb2

FLAGS = flags.FLAGS

_TRANSFORM_TMP_DIR = tempfile.gettempdir()
_TRANSFORM_METADATA_DIR = os.path.join(_TRANSFORM_TMP_DIR,
                                       'transformed_metadata')
_TRANSFORM_MODEL_DIR = os.path.join(_TRANSFORM_TMP_DIR, 'transform_fn')


def _create_input_metadata(features_config):
  """Constructs a dataset_metadata.DatasetMetadata for input metadata.

  Args:
    features_config: A dictionary of features configuration.

  Returns:
    A dataset_metadata.DatasetMetadata object.
  """
  feature_spec = {
      features_config['target_feature']: tf.io.VarLenFeature(dtype=tf.float32)
  }
  feature_spec.update({
      feature: tf.io.VarLenFeature(dtype=tf.int64)
      for feature in features_config['forward_features']
  })
  feature_spec.update({
      feature: tf.io.VarLenFeature(dtype=tf.float32)
      for feature in features_config['numeric_features']
  })
  feature_spec.update({
      feature: tf.io.VarLenFeature(dtype=tf.string)
      for feature in features_config['categorical_features']
  })
  schema = schema_utils.schema_from_feature_spec(feature_spec)
  return dataset_metadata.DatasetMetadata(schema)


def _create_output_metadata(features_config, min_value, max_value):
  """Constructs a custom dataset_metadata.DatasetMetadata for output metadata.

  Args:
    features_config: A dictionary of features configuration.
    min_value: Minimum value for IntDomain.
    max_value: Maximum value for IntDomain (for details refer to
      tensorflow_metadata/proto/v0/schema.proto).

  Returns:
    A dataset_metadata.DatasetMetadata object.
  """
  schema = {
      features_config['target_feature']:
          tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32)
  }
  schema.update({
      feature: tf.io.VarLenFeature(dtype=tf.int64)
      for feature in features_config['forward_features']
  })
  schema.update({
      utils.make_transformed_key(feature):
      tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32)
      for feature in features_config['numeric_features']
  })
  categorical_col_schema = tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64)
  categorical_col_int_domain = schema_pb2.IntDomain(
      min=min_value, max=max_value, is_categorical=True)
  domains = {}
  for feature in features_config['categorical_features']:
    name = utils.make_transformed_key(feature)
    schema[name] = categorical_col_schema
    domains[name] = categorical_col_int_domain

  return dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(schema, domains))


class TestFeatureTransformer(tft_unit.TransformTestCase):

  @classmethod
  def setUpClass(cls):
    """Sets up materialized preprocessed data for unit tests.

    The function is called once before the tests start.
    """
    super(TestFeatureTransformer, cls).setUpClass()
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    args = argparse.Namespace(
        runner='DirectRunner',
        features_config=os.path.join(
            os.path.dirname(__file__), 'features_config.cfg'),
        project_id='test_project',
        region='asia-northeast1',
        job_name='test_job',
        all_data=os.path.join(test_data_dir, 'data_all.csv'),
        train_data=os.path.join(test_data_dir, 'data_train.csv'),
        eval_data=os.path.join(test_data_dir, 'data_eval.csv'),
        predict_data=os.path.join(test_data_dir, 'data_predict.csv'),
        data_source='csv',
        skip_header_lines=1,
        mode_train=True,
        schema_file='schema.pbtxt',
        transform_dir=_TRANSFORM_TMP_DIR,
        output_dir=_TRANSFORM_TMP_DIR)
    task.main(args)

  @classmethod
  def tearDownClass(cls):
    super(TestFeatureTransformer, cls).tearDownClass()
    shutil.rmtree(_TRANSFORM_TMP_DIR, ignore_errors=True)

  @tft_unit.named_parameters(('WithDeepCopy', True))
  def test_preprocessing_fn(self, with_deep_copy):
    """Tests if transformed result matches expected data and metadata."""
    features_config = {
        'target_feature': 't',
        'forward_features': ['i', 'n3'],
        'numeric_features': ['n1', 'n2'],
        'categorical_features': ['c1', 'c2'],
        'oov_size': 5,
        'vocab_size': 10
    }
    input_metadata = _create_input_metadata(features_config)
    input_data = [{
        't': [0.0],
        'i': [0],
        'n1': [1.0],
        'n2': [2.0],
        'n3': [1],
        'c1': ['test1'],
        'c2': ['test2']
    }, {
        't': [1.0],
        'i': [1],
        'n1': [3.0],
        'n2': [4.0],
        'n3': [3],
        'c1': ['test2'],
        'c2': ['test1']
    }]
    expected_data = [{
        't': [0.0],
        'i': [0],
        'n3': [1],
        'tr_n1': [-1.0],
        'tr_n2': [-1.0],
        'tr_c1': [1],
        'tr_c2': [0]
    }, {
        't': [1.0],
        'i': [1],
        'n3': [3],
        'tr_n1': [1.0],
        'tr_n2': [1.0],
        'tr_c1': [0],
        'tr_c2': [1]
    }]
    expected_metadata = _create_output_metadata(features_config, 0, 6)
    preprocessing_fn = task._make_preprocessing_fn(features_config)
    with tft_beam.Context(use_deep_copy_optimization=with_deep_copy):
      self.assertAnalyzeAndTransformResults(
          input_data=input_data,
          input_metadata=input_metadata,
          preprocessing_fn=preprocessing_fn,
          expected_data=expected_data,
          expected_metadata=expected_metadata)

  def test_transform_model_variables_empty(self):
    """Test if there are no files on `transform_fn/variables` directory.

    Tensorflow graph should be `frozen` and saved as .pb file and checkpoint
    files (.meta, .data) should not exist.
    """
    self.assertFalse(
        os.listdir(os.path.join(_TRANSFORM_MODEL_DIR, 'variables')))

  @parameterized.parameterized.expand([
      (_TRANSFORM_MODEL_DIR, 'saved_model.pb'),
      (_TRANSFORM_METADATA_DIR, 'schema.pbtxt'),
  ])
  def test_transform_model_and_schema(self, created_dir, created_file):
    """Test if transformer model and schema proto file are created."""
    self.assertTrue(os.path.isdir(created_dir))
    self.assertTrue(os.path.isfile(os.path.join(created_dir, created_file)))

  @parameterized.parameterized.expand([
      ('train*.tfrecord'),
      ('eval*.tfrecord'),
      ('predict*.tfrecord'),
  ])
  def test_transform_train_eval_predict_files(self, transformed_file):
    """Test if transformed files for train/eval/predict have been created."""
    self.assertTrue(
        glob.glob(os.path.join(_TRANSFORM_TMP_DIR, transformed_file)))


if __name__ == '__main__':
  tft_unit.test_case.main()
