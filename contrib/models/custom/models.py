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
"""Template module for Regression and Classification models."""

import enum
import os
from typing import Any, Callable, Dict, List, Text, Tuple
import tensorflow as tf

from driblet.contrib.models.custom import input_functions

_FORWARD_FEATURES = 'forward_features'
logger = tf.get_logger()


@enum.unique
class EstimatorType(enum.Enum):
  """Tf.Estimator types."""
  REGRESSOR = 'Regressor'
  COMBINED_REGRESSOR = 'CombinedRegressor'
  CLASSIFIER = 'Classifier'
  COMBINED_CLASSIFIER = 'CombinedClassifier'


def custom_model_fn(
    estimator: tf.estimator.Estimator, forward_features: List[Text],
    estimator_type: Text, include_prediction_class: bool,
    probability_output_key: Text, prediction_output_key: Text
) -> Callable[[Dict[Text, Any], Dict[Text, Any], Text],
              tf.estimator.EstimatorSpec]:
  """Creates a custom model_fn for tf.estimator.Estimator.

  Args:
    estimator: Instance of `tf.estimator.Estimator`.
    forward_features: Features to forward during prediction time.
    estimator_type: Type of the estimator. Should be one of {Regressor,
      CombinedRegressor, Classifier, CombinedClassifier}.
      include_prediction_class: If set True, classification prediction output
        will include predicted classes.
      probability_output_key: Key name for output probability value.
      prediction_output_key: Key name for output prediction value.

  Returns:
    model_fn: A  function to be called by `tf.estimator.Estimator`
          object to construct the graph.
  """

  def model_fn(features: Dict[Text, Any], labels: Dict[Text, Any],
               mode: Text) -> tf.estimator.EstimatorSpec:
    """Extends model_fn to include custom output fields.

    Args:
      features: Input features.
      labels: Input labels.
      mode: One of tf.estimator.ModeKeys{TRAIN, EVAL, PREDICT}.

    Returns:
      estimator_spec: Ops and objects returned from a model_fn and passed to
      an Estimator.
    """
    estimator_spec = estimator.model_fn(
        features=features, labels=labels, mode=mode, config=estimator.config)
    if estimator_spec.export_outputs:
      if estimator_type in [
          EstimatorType.REGRESSOR.value, EstimatorType.COMBINED_REGRESSOR.value
      ]:
        output = {
            prediction_output_key:
                tf.squeeze(estimator_spec.predictions['predictions'])
        }
      elif estimator_type in [
          EstimatorType.CLASSIFIER.value,
          EstimatorType.COMBINED_CLASSIFIER.value
      ]:
        probabilities = estimator_spec.predictions['probabilities']
        # Output prediction probabilities only for positive class.
        # TODO(): Add support for multiclass classifiction output.
        output = {
            probability_output_key:
                tf.squeeze(tf.slice(probabilities, [0, 1], [-1, -1]))
        }
        if include_prediction_class:
          output.update({
              prediction_output_key:
                  tf.squeeze(estimator_spec.predictions['class_ids'])
          })
      else:
        raise ValueError('Estimator %s is not supported' % estimator_type)
      for feature in forward_features:
        output.update(
            {feature: tf.squeeze(estimator_spec.predictions[feature])})

      predicted_output = tf.estimator.export.PredictOutput(
          estimator_spec.predictions)
      export_output = tf.estimator.export.PredictOutput(output)
      estimator_spec.export_outputs['predict'] = predicted_output
      estimator_spec.export_outputs['serving_default'] = export_output
    return estimator_spec

  return model_fn


def calculate_rmse(
    labels: tf.Tensor,
    predictions: Dict[Text, tf.Tensor]) -> Dict[Text, Tuple[float, float]]:
  """Calculates Root Mean Squared Error for the Regression model.

  Args:
    labels: Actual label values.
    predictions: Model prediction values.

  Returns:
    rmse: Root Mean Squared Error values.
  """
  return {
      'rmse':
          tf.compat.v1.metrics.root_mean_squared_error(
              tf.cast(labels, tf.float32), predictions['predictions'])
  }


def forward_features(estimator, keys=None, sparse_default_values=None):
  """Forwards features to predictions dictionary.

  NOTE: This method is partially forked from deprecated
  tf.contrib.estimator.forward_features.

  Args:
    estimator: A `tf.estimator.Estimator` object.
    keys: A `string` or a `list` of `string`. If it is `None`, all of the
      `features` in `dict` is forwarded to the `predictions`. If it is a
      `string`, only given key is forwarded. If it is a `list` of strings, all
      the given `keys` are forwarded.
    sparse_default_values: A dict of `str` keys mapping the name of the sparse
      features to be converted to dense, to the default value to use. Only
      sparse features indicated in the dictionary are converted to dense and the
      provided default value is used.

  Returns:
      A new `tf.estimator.Estimator` which forwards features to predictions.

  Raises:
    ValueError:
      * if `keys` is already part of `predictions`.
      * if 'keys' does not exist in `features`.
  """

  def get_keys(features):
    if keys is None:
      return features.keys()
    return keys

  def new_model_fn(features, labels, mode, config):
    spec = estimator.model_fn(features, labels, mode, config)
    predictions = spec.predictions
    if predictions is None:
      return spec
    for key in get_keys(features):
      feature = tf.compat.v1.convert_to_tensor_or_sparse_tensor(features[key])
      if sparse_default_values and (key in sparse_default_values):
        if not isinstance(feature, tf.sparse.SparseTensor):
          raise ValueError(
              'Feature ({}) is expected to be a `SparseTensor`.'.format(key))
        feature = tf.sparse.to_dense(
            feature, default_value=sparse_default_values[key])
      if not isinstance(feature, tf.Tensor):
        raise ValueError(
            'Feature ({}) should be a Tensor. Please use `keys` '
            'argument of forward_features to filter unwanted features, or'
            'add key to argument `sparse_default_values`.'
            'Type of features[{}] is {}.'.format(key, key, type(feature)))
      predictions[key] = feature
    spec = spec._replace(predictions=predictions)
    if spec.export_outputs:
      for ekey in ['predict', 'serving_default']:
        if (ekey in spec.export_outputs and isinstance(
            spec.export_outputs[ekey], tf.estimator.export.PredictOutput)):
          export_outputs = spec.export_outputs[ekey].outputs
          for key in get_keys(features):
            export_outputs[key] = predictions[key]

    return spec

  return tf.estimator.Estimator(
      model_fn=new_model_fn,
      model_dir=estimator.model_dir,
      config=estimator.config)


def make_optimizer(optimizer_name: Text, learning_rate: float):
  """Provides tf.train.<optimizer> instance given optimizer name.

  Args:
    optimizer_name: Name of the DNN optimizer.
    learning_rate: Learning rate for the model

  Returns:
    optimizer: Instance of tf.train.<optimizer>.

  Raises:
    ValueError: If provided optimizer is not supported.
  """
  if optimizer_name == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
  elif optimizer_name == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer_name == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer_name == 'Adadelta':
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
  elif optimizer_name == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
  else:
    raise ValueError('Optimizer %s is not supported. Provide one of '
                     '{Adagrad, Adam, SGD, Adadelta, RMSProp}' % optimizer_name)

  return optimizer


def create_estimator(features_config: Dict[Text, Any], job_dir: Text,
                     first_layer_size: int, num_layers: int,
                     estimator_type: Text, linear_optimizer: Text,
                     dnn_optimizer: Text, save_checkpoints_steps: int,
                     keep_checkpoint_max: int, dnn_dropout: float,
                     learning_rate: float, include_prediction_class: bool,
                     probability_output_key: Text,
                     prediction_output_key: Text) -> tf.estimator.Estimator:
  """Creates `tf.estimator.Estimator` for training and evaluation.

  Args:
    features_config: Features' configuration.
    job_dir: GCS location to write checkpoints and export models.
    first_layer_size: Size of the first layer.
    num_layers: Number of NN layers.
    estimator_type: Type of the estimator. Should be one of {Regressor,
      CombinedRegressor, Classifier, CombinedClassifier}.
    linear_optimizer: Optimizer for Linear model.
    dnn_optimizer: Optimizer for DNN model.
    save_checkpoints_steps: Save checkpoints every N steps.
    keep_checkpoint_max: Maximum number of recent checkpoint files to keep.
    dnn_dropout: The probability to drop out a given unit in DNN.
    learning_rate: Learning rate for the model.
    include_prediction_class: If set True, classification prediction output will
      include predicted classes.
    probability_output_key: Key name for output probability value.
    prediction_output_key: Key name for output prediction value.

  Returns:
    estimator: tf.estimator.Estimator.
  """
  num_buckets = features_config['vocab_size'] + features_config['oov_size']
  linear_features, dense_features = input_functions.create_feature_columns(
      features_config['categorical_features'],
      features_config['numeric_features'], num_buckets, estimator_type)
  run_config = tf.estimator.RunConfig(
      model_dir=os.path.expanduser(job_dir),
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=keep_checkpoint_max)
  # `hidden_units` for DNN are constructed based on pow(2, i) or 2**i
  # Ex: if first_layer_size is 10 and num_layers is 3, `hidden_units` that DNN
  # receives will be [10, 5, 2], where 10 is number of neurons in first layer,
  # 5 is size of hidden layer neurons, 2 is size of output layer neurons.
  hidden_units = []
  for i in range(int(num_layers)):
    hidden_units.append(max(int(first_layer_size / (pow(2, i))), 2))
  optimizer = make_optimizer(dnn_optimizer, learning_rate)

  if estimator_type == EstimatorType.REGRESSOR.value:
    logger.info('Running %s estimator.', EstimatorType.REGRESSOR.name)
    estimator = tf.estimator.DNNRegressor(
        config=run_config,
        optimizer=optimizer,
        feature_columns=dense_features,
        hidden_units=hidden_units,
        dropout=dnn_dropout)
  elif estimator_type == EstimatorType.COMBINED_REGRESSOR.value:
    logger.info('Running %s estimator.', EstimatorType.COMBINED_REGRESSOR.name)
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        config=run_config,
        linear_feature_columns=linear_features,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dense_features,
        dnn_optimizer=optimizer,
        dnn_hidden_units=hidden_units,
        dnn_dropout=dnn_dropout)
  elif estimator_type == EstimatorType.CLASSIFIER.value:
    logger.info('Running %s estimator.', EstimatorType.CLASSIFIER.name)
    estimator = tf.estimator.DNNClassifier(
        config=run_config,
        optimizer=optimizer,
        feature_columns=dense_features,
        hidden_units=hidden_units,
        dropout=dnn_dropout)
  elif estimator_type == EstimatorType.COMBINED_CLASSIFIER.value:
    logger.info('Running %s estimator.', EstimatorType.COMBINED_CLASSIFIER.name)
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        config=run_config,
        linear_feature_columns=linear_features,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dense_features,
        dnn_optimizer=optimizer,
        dnn_hidden_units=hidden_units,
        dnn_dropout=dnn_dropout)
  else:
    raise ValueError(
        '%s is not supported. Please, choose one of '
        '{Regressor, CombinedRegressor, Classifier, CombinedClassifier}' %
        estimator_type)

  estimator = forward_features(estimator, features_config[_FORWARD_FEATURES])

  # Add RMSE metric if estimator is one of Regression models.
  if estimator_type in [
      EstimatorType.REGRESSOR.value, EstimatorType.COMBINED_REGRESSOR.value
  ]:
    estimator = tf.estimator.add_metrics(estimator, calculate_rmse)
  model_fn = custom_model_fn(
      estimator=estimator,
      forward_features=features_config[_FORWARD_FEATURES],
      estimator_type=estimator_type,
      include_prediction_class=include_prediction_class,
      probability_output_key=probability_output_key,
      prediction_output_key=prediction_output_key)
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=os.path.expanduser(job_dir),
      config=run_config)
  return estimator
