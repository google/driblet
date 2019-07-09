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
"""Module for training and evaluating model."""

import os

import tensorflow as tf
from trainer import input_functions
from workflow.dags.tasks.preprocess import features_config


def custom_model_fn(estimator):
  """Returns custom model_fn for `tf.estimator.Estimator`.

  Args:
    estimator: Instance of `tf.estimator.Estimator`.

  Returns:
    model_fn: This is a function that is  called by `tf.estimator.Estimator`
          object to construct the graph.
  """

  def model_fn(features, labels, mode):
    """Extends model_fn to include custom output fields.

    Args:
      features: Dictionary of input features.
      labels: Dictionary of input labels.
      mode: One of tf.estimator.ModeKeys{TRAIN, EVAL, PREDICT}.

    Returns:
      estimator_spec: Ops and objects returned from a model_fn and passed to an
      Estimator.
    """
    estimator_spec = estimator.model_fn(
        features=features, labels=labels, mode=mode, config=estimator.config)
    if estimator_spec.export_outputs:
      output = {
          'id':
              tf.squeeze(estimator_spec.predictions[features_config.ID_FEATURE]
                        ),
          'prediction':
              tf.squeeze(estimator_spec.predictions['class_ids']),
          'probability':
              tf.reduce_max(
                  estimator_spec.predictions['probabilities'], axis=1)
      }
      predicted_output = tf.estimator.export.PredictOutput(
          estimator_spec.predictions)
      export_output = tf.estimator.export.PredictOutput(output)
      estimator_spec.export_outputs['predict'] = predicted_output
      estimator_spec.export_outputs['serving_default'] = export_output
    return estimator_spec

  return model_fn


def create_estimator(hparams):
  """Creates `tf.estimator.Estimator` for training and evaluation.

  Args:
    hparams: A TensorFlow `HParams` object to provide hyperparameters to the
      model.

  Returns:
    estimator: `tf.estimator.Estimator`
  """
  num_buckets = features_config.VOCAB_SIZE + features_config.OOV_SIZE
  linear_features, dense_features = input_functions.create_feature_columns(
      features_config.CATEGORICAL_FEATURES, features_config.NUMERIC_FEATURES,
      num_buckets)
  run_config = tf.estimator.RunConfig(
      model_dir=os.path.expanduser(hparams.job_dir),
      save_checkpoints_steps=hparams.save_checkpoints_steps,
      keep_checkpoint_max=hparams.keep_checkpoint_max)
  # hidden_units for DNN are constructed based on pow(2, i) or 2**i
  # Ex: if first_layer_size is 10 and num_layers is 3, hidden_units that DNN
  # receives will be [10, 5, 2], where 10 is number of neurons in first layer,
  # 5 is size of hidden layer neurons, 2 is size of output layer neurons
  hidden_units = []
  for i in range(int(hparams.num_layers)):
    hidden_units.append(max(int(hparams.first_layer_size / (pow(2, i))), 2))
  estimator = tf.estimator.DNNLinearCombinedClassifier(
      config=run_config,
      linear_feature_columns=linear_features,
      linear_optimizer=hparams.linear_optimizer,
      dnn_dropout=hparams.dnn_dropout,
      dnn_feature_columns=dense_features,
      dnn_hidden_units=hidden_units,
      dnn_optimizer=hparams.dnn_optimizer)

  estimator = tf.contrib.estimator.forward_features(
      estimator, features_config.FORWARD_FEATURE)
  estimator = tf.estimator.Estimator(
      model_fn=custom_model_fn(estimator),
      model_dir=os.path.expanduser(hparams.job_dir),
      config=run_config)
  return estimator
