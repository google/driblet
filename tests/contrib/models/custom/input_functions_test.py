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

"""Tests for driblet.models.custom.input_functions."""

import os
import shutil
import tempfile

import numpy as np
import parameterized
import tensorflow.compat.v1 as tf
from tensorflow_transform.saved import saved_transform_io

from driblet.contrib.models.custom import input_functions

TEST_FEATURE_ID = 'id'
TEST_TARGET_FEATURE = 'test_target'
TEST_FEATURE = 'test_feature'
TEST_FEATURE_ID_VALUE = [1]
TEST_TARGET_FEATURE_VALUE = [1]
TEST_FEATURE_VALUE = [1.0]
TEST_DATA_FILE = 'test_data.tfrecord'
TEST_BUCKET_SIZE = 2
TEST_METADATA_SCHEMA = """
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
""" % (TEST_FEATURE_ID, TEST_TARGET_FEATURE, TEST_FEATURE)


def _create_test_data():
  """Creates serialized test data in tf.Example format.

  Returns:
    Serialized tf.Example proto.
  """
  feature = {
      TEST_FEATURE_ID:
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=TEST_FEATURE_ID_VALUE)),
      TEST_TARGET_FEATURE:
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=TEST_TARGET_FEATURE_VALUE)),
      TEST_FEATURE:
          tf.train.Feature(
              float_list=tf.train.FloatList(value=TEST_FEATURE_VALUE))
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
  tf.gfile.MakeDirs(test_transform_dir)
  with open(test_schema, 'w') as schema_file:
    schema_file.write(TEST_METADATA_SCHEMA)


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
          TEST_FEATURE_ID: tf.placeholder(tf.int64, shape=[1]),
          TEST_FEATURE: input_placeholder
      }
      output_dict = {
          TEST_FEATURE_ID:
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
    super(InputFunctionsTest, cls).tearDownClass()
    shutil.rmtree(cls.tempdir)

  def test_convert_sparse_to_dense_provides_dense_tensor(self):
    """Tests that SparseTensor is correctly converted to dense Tensor."""
    input_tensor = tf.SparseTensor(indices=[[0]], values=[1], dense_shape=[1])

    actual = input_functions.convert_sparse_to_dense(input_tensor)

    self.assertShapeEqual(np.array([1]), actual)
    self.assertEqual(actual.dtype, tf.int32)
    self.assertIsInstance(actual, tf.Tensor)

  @parameterized.parameterized.expand([
      ('CombinedRegressor'),
      ('CombinedClassifier'),
  ])
  def test_create_feature_columns_creates_transformer_keys(
      self, estimator_type):
    test_categorical_features = ['c1']
    test_numeric_features = ['n1']
    expected_keys = ['tr_c1', 'tr_n1']

    actual_keys = []
    actual_linear_features, actual_dense_features = (
        input_functions.create_feature_columns(test_categorical_features,
                                               test_numeric_features,
                                               TEST_BUCKET_SIZE,
                                               estimator_type))

    for linear_feature, dense_feature in zip(actual_linear_features,
                                             actual_dense_features):
      actual_keys.append(linear_feature.key)
      actual_keys.append(dense_feature.key)
      # TODO(zmtbnv): Currently, feature_column module is not visible to
      # this package (http://go/jizas). Implement assertIsInstance to verify
      # if features are instances of feature_column._IdentityCategoricalColumn
      # and feature_column._NumericColumn.
    self.assertListEqual(actual_keys, expected_keys)

  @parameterized.parameterized.expand([
      ('Regressor'),
      ('Classifier'),
  ])
  def test_create_feature_columns_provides_embedding_dimension(
      self, estimator_type):
    test_categorical_features = ['c1']
    test_numeric_features = ['n1']

    expected_dimension = int(6 * TEST_BUCKET_SIZE**0.25)
    actual_linear_features, _ = (
        input_functions.create_feature_columns(test_categorical_features,
                                               test_numeric_features,
                                               TEST_BUCKET_SIZE,
                                               estimator_type))

    for linear_feature in actual_linear_features:
      self.assertEqual(linear_feature.dimension, expected_dimension)

  @parameterized.parameterized.expand([
      ('Regressor', 5, 0),
      ('Classifier', 5, 0),
      ('CombinedRegressor', 3, 2),
      ('CombinedClassifier', 3, 2),
  ])
  def test_create_feature_columns_provides_correct_feature_lists(
      self, estimator_type, expected_dense_feature_length,
      expected_linear_feature_length):
    test_categorical_features = ['c1', 'c2']
    test_numeric_features = ['n1', 'n2', 'n3']

    actual_linear_features, actual_dense_features = (
        input_functions.create_feature_columns(test_categorical_features,
                                               test_numeric_features,
                                               TEST_BUCKET_SIZE,
                                               estimator_type))
    self.assertEqual(len(actual_dense_features), expected_dense_feature_length)
    self.assertEqual(
        len(actual_linear_features), expected_linear_feature_length)

  def test_get_input_fn_povides_correct_features_target_values(self):
    testfile = os.path.join(self.tempdir, TEST_DATA_FILE)
    _write_test_data_to_disk(testfile)

    input_fn = input_functions.get_input_fn(
        filename_patterns=[testfile],
        tf_transform_dir=self.tempdir,
        target_feature=TEST_TARGET_FEATURE,
        forward_features=[TEST_FEATURE_ID],
        num_epochs=1,
        batch_size=1)
    features, target = input_fn()

    with self.session() as session:
      features, target = session.run((features, target))
    self.assertEqual(target, TEST_TARGET_FEATURE_VALUE)
    self.assertEqual(features[TEST_FEATURE], TEST_FEATURE_VALUE)
    self.assertDTypeEqual(features[TEST_FEATURE], np.float32)
    self.assertDTypeEqual(target, np.int64)

  def test_example_serving_receiver_fn(self):
    _write_schema_to_disk(self.tempdir)
    _create_and_write_test_saved_model(self.tempdir)
    raw_feature_spec = {TEST_FEATURE: tf.FixedLenFeature([], tf.float32)}
    expected_feature_keys = [TEST_FEATURE_ID, 'test_scaled_feature']

    actual = input_functions.example_serving_receiver_fn(
        self.tempdir, raw_feature_spec, TEST_TARGET_FEATURE, [TEST_FEATURE_ID])
    actual_keys = sorted(actual.features.keys())

    self.assertIsInstance(actual, tf.estimator.export.ServingInputReceiver)
    self.assertListEqual(actual_keys, expected_feature_keys)


if __name__ == '__main__':
  tf.test.main()
