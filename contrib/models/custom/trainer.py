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
"""Module to run training and evaluation."""

import argparse
import os

import tensorflow as tf
from driblet.contrib.models.custom import input_functions
from driblet.contrib.models.custom import models
from driblet.contrib.models.custom.feature_transformer import utils

_FORWARD_FEATURES = 'forward_features'
_TARGET_FEATURE = 'target_feature'
_FEATURES_CONFIG_FILE = '/tmp/features_config.cfg'


def parse_arguments():
  """Initialize command line parser using arparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--features_config_file',
      default=None,
      help='Path to features configuration file (.cfg)',
      type=str)

  # Flags related to input data.
  parser.add_argument(
      '--train_data',
      default=None,
      help='GCS or local paths to training data',
      nargs='+',
      type=str)
  parser.add_argument(
      '--eval_data',
      default=None,
      help='GCS or local paths to evaluation data',
      nargs='+',
      type=str)
  parser.add_argument(
      '--schema_file',
      default=None,
      help='File holding the schema for the input data',
      type=str)

  # Flags related to feature transformation.
  parser.add_argument(
      '--include_prediction_class',
      default=False,
      help='If set True, classification prediction output will include predicted classes.',
      type=bool)

  parser.add_argument(
      '--probability_output_key',
      default='probability',
      help='Key name for output probability value.',
      type=str)

  parser.add_argument(
      '--prediction_output_key',
      default='prediction',
      help='Key name for output prediction value.',
      type=str)

  # Flags related to model's output.
  parser.add_argument(
      '--transform_dir',
      default=None,
      help='Tf-transform directory with model from preprocessing step',
      type=str)
  # Flags related to training hyperparameters.
  parser.add_argument(
      '--job-dir',
      default=None,
      help='GCS location to write checkpoints and export models.',
      type=str)
  parser.add_argument(
      '--model_name', default=None, help='Name of the model to save.', type=str)
  parser.add_argument(
      '--estimator_type',
      default='Regressor',
      help='Type of the estimator. Should be one of [Regressor, '
      'CombinedRegressor, Classifier, CombinedClassifier].',
      type=str)
  parser.add_argument(
      '--train_steps',
      default=1000,
      help='Count of steps to run the training job for',
      type=int)
  parser.add_argument(
      '--train_batch_size', default=100, help='Train batch size.', type=int)
  parser.add_argument(
      '--eval_steps',
      default=100,
      help='Number of steps to run evaluation for at each checkpoint',
      type=int)
  parser.add_argument(
      '--eval_batch_size', default=50, help='Eval batch size.', type=int)
  parser.add_argument(
      '--num_epochs', default=1, help='Number of epochs.', type=int)
  parser.add_argument(
      '--first_layer_size',
      default=10,
      help='Size of the first layer.',
      type=int)
  parser.add_argument(
      '--num_layers', default=2, help='Number of NN layers.', type=int)
  parser.add_argument(
      '--save_checkpoints_steps',
      default=100,
      help='Save checkpoints every N steps.',
      type=int)
  parser.add_argument(
      '--keep_checkpoint_max',
      default=3,
      help='Maximum number of recent checkpoint files to keep.',
      type=int)
  parser.add_argument(
      '--exports_to_keep',
      default=1,
      help='Number of model exports to keep.',
      type=int)
  parser.add_argument(
      '--start_delay_secs',
      default=1,
      help='Start evaluating after N seconds.',
      type=int)
  parser.add_argument(
      '--throttle_secs', default=2, help='Evaluate every N seconds.', type=int)
  parser.add_argument(
      '--dnn_optimizer',
      default='Adam',
      help='Optimizer for DNN model.',
      type=str)
  parser.add_argument(
      '--dnn_dropout', default=0.1, help='Dropout value for DNN.', type=float)
  parser.add_argument(
      '--linear_optimizer',
      default='Ftrl',
      help='Optimizer for linear model.',
      type=str)
  parser.add_argument(
      '--learning_rate',
      default=0.001,
      help='Learning rate for the model.',
      type=float)
  return parser.parse_args()


def train_and_evaluate(hparams) -> None:
  """Trains and evaluates the model.

  Args:
    hparams: An instance of HParams object describing the hyper-parameters for
      the model.

  Raises:
    RuntimeError: When features config file does not exist.
  """

  config_path = hparams.features_config_file
  if config_path.startswith('gs://'):
    config_path = _FEATURES_CONFIG_FILE
    tf.io.gfile.copy(hparams.features_config_file, config_path, overwrite=True)

  if not os.path.isfile(config_path):
    raise RuntimeError('Features config `{}` not exist.'.format(config_path))

  features_config = utils.parse_features_config(config_path)
  train_input_fn = input_functions.get_input_fn(
      filename_patterns=hparams.train_data,
      tf_transform_dir=hparams.transform_dir,
      target_feature=features_config[_TARGET_FEATURE],
      forward_features=features_config[_FORWARD_FEATURES],
      num_epochs=hparams.num_epochs,
      batch_size=hparams.train_batch_size)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=hparams.train_steps)

  eval_input_fn = input_functions.get_input_fn(
      filename_patterns=hparams.eval_data,
      tf_transform_dir=hparams.transform_dir,
      target_feature=features_config[_TARGET_FEATURE],
      forward_features=features_config[_FORWARD_FEATURES],
      num_epochs=hparams.num_epochs,
      batch_size=hparams.eval_batch_size)
  schema_file = utils.read_schema(hparams.schema_file)
  raw_feature_spec = utils.get_raw_feature_spec(schema_file,
                                                tf.estimator.ModeKeys.TRAIN)

  serving_receiver_fn = lambda: input_functions.example_serving_receiver_fn(
      hparams.transform_dir, raw_feature_spec, features_config[_TARGET_FEATURE],
      features_config[_FORWARD_FEATURES])
  exporter = tf.estimator.LatestExporter(
      name=hparams.model_name,
      serving_input_receiver_fn=serving_receiver_fn,
      exports_to_keep=hparams.exports_to_keep)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hparams.eval_steps,
      start_delay_secs=hparams.start_delay_secs,
      throttle_secs=hparams.throttle_secs,
      exporters=[exporter])

  estimator = models.create_estimator(
      features_config=features_config,
      job_dir=hparams.job_dir,
      first_layer_size=hparams.first_layer_size,
      num_layers=hparams.num_layers,
      estimator_type=hparams.estimator_type,
      linear_optimizer=hparams.linear_optimizer,
      dnn_optimizer=hparams.dnn_optimizer,
      dnn_dropout=hparams.dnn_dropout,
      learning_rate=hparams.learning_rate,
      save_checkpoints_steps=hparams.save_checkpoints_steps,
      keep_checkpoint_max=hparams.keep_checkpoint_max,
      include_prediction_class=hparams.include_prediction_class,
      probability_output_key=hparams.probability_output_key,
      prediction_output_key=hparams.prediction_output_key)
  tf.estimator.train_and_evaluate(
      estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def main() -> None:
  """Main method to call train and evaluate."""
  tf.get_logger().setLevel('ERROR')
  args = parse_arguments()
  train_and_evaluate(args)


if __name__ == '__main__':
  main()
