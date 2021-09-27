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

"""Tests for driblet.models.custom."""

from typing import Any

import mock
import parameterized
import tensorflow as tf

from driblet.contrib.models.custom import models
# Common keys used for testing.
ID_COL = 'id_col'
PROBABILITY_OUTPUT_KEY = 'probabilities'
PREDICTION_OUTPUT_KEY = 'predictions'

# Test values for Regression prediction output.
REGRESSOR_OUTPUT_VALUES = {
    ID_COL: tf.constant([[1]]),
    PREDICTION_OUTPUT_KEY: tf.constant([[1]])
}

# Test values for Classification prediction output.
CLASSIFIER_OUTPUT_VALUES = {
    ID_COL: tf.constant([[1]]),
    'class_ids': tf.constant([[1]]),
    'classes': tf.constant([[1]]),
    'logistic': tf.constant([[0.3]]),
    'logits': tf.constant([[0.2]]),
    PROBABILITY_OUTPUT_KEY: tf.constant([[0.5], [1.0]])
}


class ModelsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_estimator = mock.MagicMock(spec=tf.estimator.Estimator)
    self.mock_model_fn = self.mock_estimator.model_fn()
    self.mock_export_output = {'predict': Any, 'serving_default': Any}

  @parameterized.parameterized.expand([
      (REGRESSOR_OUTPUT_VALUES, 'Regressor', 2),
      (REGRESSOR_OUTPUT_VALUES, 'CombinedRegressor', 2),
      (CLASSIFIER_OUTPUT_VALUES, 'Classifier', 3),
      (CLASSIFIER_OUTPUT_VALUES, 'CombinedClassifier', 3)
  ])
  def test_custom_model_fn_default_values(self, output_values, estimator_type,
                                          output_length):
    """Tests that custom model_fn provides correct export outputs."""
    type(self.mock_model_fn).predictions = mock.PropertyMock(
        return_value=output_values)
    type(self.mock_model_fn).export_outputs = mock.PropertyMock(
        return_value=self.mock_export_output)
    custom_model_fn = models.custom_model_fn(
        estimator=self.mock_estimator,
        forward_features=[ID_COL],
        estimator_type=estimator_type,
        include_prediction_class=True,
        probability_output_key=PROBABILITY_OUTPUT_KEY,
        prediction_output_key=PREDICTION_OUTPUT_KEY)
    estimator_spec = custom_model_fn(None, None, tf.estimator.ModeKeys.TRAIN)

    actual_predict = estimator_spec.export_outputs['predict']
    actual_serving = estimator_spec.export_outputs['serving_default']

    # Assert that both exported `predict` and `serving_default` classes
    # are instances of Tensorflow export PredictOutput class.
    self.assertIsInstance(actual_predict, tf.estimator.export.PredictOutput)
    self.assertIsInstance(actual_serving, tf.estimator.export.PredictOutput)

    # Assert that export_output have correct lengths.
    # Regression model should have serving output length of 2: id & prediction.
    # Classification model should have output length of 3: id, prediction &
    # probability.
    self.assertEqual(len(actual_predict.outputs), len(output_values))
    self.assertEqual(len(actual_serving.outputs), output_length)

  @parameterized.parameterized.expand([
      (CLASSIFIER_OUTPUT_VALUES, 'Classifier', True, 'probability_key1',
       'prediction_key1'),
      (CLASSIFIER_OUTPUT_VALUES, 'CombinedClassifier', False,
       'probability_key2', 'prediction_key2')
  ])
  def test_custom_model_fn_classifier_outputs_custom_keys(
      self, output_values, estimator_type, include_prediction_class,
      probability_output_key, prediction_output_key):
    type(self.mock_model_fn).predictions = mock.PropertyMock(
        return_value=output_values)
    type(self.mock_model_fn).export_outputs = mock.PropertyMock(
        return_value=self.mock_export_output)
    custom_model_fn = models.custom_model_fn(
        estimator=self.mock_estimator,
        forward_features=[ID_COL],
        estimator_type=estimator_type,
        include_prediction_class=include_prediction_class,
        probability_output_key=probability_output_key,
        prediction_output_key=prediction_output_key)
    estimator_spec = custom_model_fn(None, None, tf.estimator.ModeKeys.TRAIN)

    if include_prediction_class:
      expected_output_keys = [
          probability_output_key, prediction_output_key, ID_COL
      ]
    else:
      expected_output_keys = [probability_output_key, ID_COL]

    actual_serving = estimator_spec.export_outputs['serving_default']

    self.assertListEqual([*actual_serving.outputs], expected_output_keys)

  @parameterized.parameterized.expand([
      (REGRESSOR_OUTPUT_VALUES, 'Regressor', [ID_COL]),
      (REGRESSOR_OUTPUT_VALUES, 'CombinedRegressor', []),
      (CLASSIFIER_OUTPUT_VALUES, 'Classifier', [ID_COL]),
      (CLASSIFIER_OUTPUT_VALUES, 'CombinedClassifier', [])
  ])
  def test_custom_model_fn_forwards_features(self, output_values,
                                             estimator_type, forward_features):
    type(self.mock_model_fn).predictions = mock.PropertyMock(
        return_value=output_values)
    type(self.mock_model_fn).export_outputs = mock.PropertyMock(
        return_value=self.mock_export_output)
    custom_model_fn = models.custom_model_fn(
        estimator=self.mock_estimator,
        forward_features=forward_features,
        estimator_type=estimator_type,
        include_prediction_class=True,
        probability_output_key=PROBABILITY_OUTPUT_KEY,
        prediction_output_key=PREDICTION_OUTPUT_KEY)
    estimator_spec = custom_model_fn(None, None, tf.estimator.ModeKeys.TRAIN)

    if estimator_type in [
        models.EstimatorType.REGRESSOR.value,
        models.EstimatorType.COMBINED_REGRESSOR.value
    ]:
      expected_output_keys = [PREDICTION_OUTPUT_KEY] + forward_features
    else:
      expected_output_keys = [PROBABILITY_OUTPUT_KEY, PREDICTION_OUTPUT_KEY
                             ] + forward_features

    actual_serving = estimator_spec.export_outputs['serving_default']

    self.assertListEqual([*actual_serving.outputs], expected_output_keys)

  def test_calculate_rmse(self):
    """Tests that RMSE is calculated correctly for given predictions and labels."""
    test_input_predictions = {'predictions': tf.constant([1.0, 2.0, 3.0, 4.0])}
    test_input_labels = tf.constant([1.0, 1.0, 5.0, 3.0])
    expected_rmse = {'rmse': (0.0, 1.2247449)}

    with self.test_session() as sess:
      actual_rmse = models.calculate_rmse(test_input_labels,
                                          test_input_predictions)
      sess.run(tf.compat.v1.local_variables_initializer())
      actual_rmse = sess.run(actual_rmse)

    self.assertEqual(actual_rmse.__repr__(), expected_rmse.__repr__())

  @parameterized.parameterized.expand([
      ('Adagrad', tf.keras.optimizers.Adagrad),
      ('Adam', tf.keras.optimizers.Adam),
      ('SGD', tf.keras.optimizers.SGD),
      ('Adadelta', tf.keras.optimizers.Adadelta),
      ('RMSprop', tf.keras.optimizers.RMSprop),
  ])
  def test_make_optimizer_creates_optimizer_instance(self, optimizer_name,
                                                     optimizer_instance):
    actual_optimizer = models.make_optimizer(optimizer_name, 0.01)

    self.assertIsInstance(actual_optimizer, optimizer_instance)

  def test_make_optimizer_raises_value_error(self):
    with self.assertRaises(ValueError):
      models.make_optimizer('unknown_optimizer', 0.01)

  @parameterized.parameterized.expand([
      ('Regressor'),
      ('Classifier'),
      ('CombinedRegressor'),
      ('CombinedClassifier'),
  ])
  def test_create_estimator(self, estimator_type):
    """Tests that Tensorfow Estimator is created correctly."""
    test_features_config = {
        'vocab_size': 4,
        'oov_size': 2,
        'forward_features': ['id_col', 'num_col1'],
        'categorical_features': ['cat_col1'],
        'numeric_features': ['num_col2']
    }
    test_job_dir = 'test_job_dir'
    test_first_layer_size = 6
    test_num_layers = 3
    test_linear_optimizer = 'Ftrl'
    test_dnn_optimizer = 'Adam'
    test_save_checkpoints_steps = 1
    test_keep_checkpoint_max = 1
    dnn_dropout = 0.1
    learning_rate = 0.001
    include_prediction_class = True
    probability_output_key = PROBABILITY_OUTPUT_KEY
    prediction_output_key = PREDICTION_OUTPUT_KEY

    actual_estimator = models.create_estimator(
        test_features_config, test_job_dir, test_first_layer_size,
        test_num_layers, estimator_type, test_linear_optimizer,
        test_dnn_optimizer, test_save_checkpoints_steps,
        test_keep_checkpoint_max, dnn_dropout, learning_rate,
        include_prediction_class, probability_output_key, prediction_output_key)

    self.assertIsInstance(actual_estimator, tf.estimator.Estimator)


if __name__ == '__main__':
  tf.test.main()
