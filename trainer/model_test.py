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
"""Tests for trainer.model."""

import mock
import model
import tensorflow as tf


class ModelTest(tf.test.TestCase):

  def test_custom_model_fn(self):
    """Tests that Tensorflow estimator spec is created correctly."""
    mock_estimator = tf.test.mock.MagicMock(spec=tf.estimator.Estimator)
    mock_model_fn = mock_estimator.model_fn()
    # Set fake values to DNNLinearCombinedClassifier prediction output
    predict_output_values = {
        'id_col': tf.constant([[1]]),
        'class_ids': tf.constant([[1]]),
        'classes': tf.constant([[1]]),
        'logistic': tf.constant([[0.3]]),
        'logits': tf.constant([[0.2]]),
        'probabilities': tf.constant([[0.5], [1.0]])
    }
    mock_export_output = {
        'predict': tf.test.mock.ANY,
        'serving_default': tf.test.mock.ANY
    }
    type(mock_model_fn).predictions = mock.PropertyMock(
        return_value=predict_output_values)
    type(mock_model_fn).export_outputs = mock.PropertyMock(
        return_value=mock_export_output)
    custom_model_fn = model.custom_model_fn(mock_estimator)
    estimator_spec = custom_model_fn(None, None, tf.estimator.ModeKeys.TRAIN)

    actual_predict = estimator_spec.export_outputs['predict']
    actual_serving = estimator_spec.export_outputs['serving_default']

    # Assert that both exported `predict` and `serving_default` classes
    # are instances of Tensorflow export PredictOutput class
    self.assertIsInstance(actual_predict, tf.estimator.export.PredictOutput)
    self.assertIsInstance(actual_serving, tf.estimator.export.PredictOutput)

    # Assert that export_output  have correct lengths
    self.assertEqual(len(actual_predict.outputs), len(predict_output_values))
    self.assertEqual(len(actual_serving.outputs), 3)

  def test_create_estimator(self):
    """Tests that Tensorfow estimator is created correctly."""
    # Hyperparameters to create the Estimator
    hparams = tf.contrib.training.HParams(
        job_dir='test_dir',
        save_checkpoints_steps=1,
        keep_checkpoint_max=1,
        num_layers=2,
        dnn_dropout=0.7,
        dnn_optimizer='test_optimizer',
        linear_optimizer='test_optimizer',
        first_layer_size=10)
    estimator = model.create_estimator(hparams)
    self.assertIsInstance(estimator, tf.estimator.Estimator)


if __name__ == '__main__':
  tf.test.main()
