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
"""Module to run training and evaluation."""

import argparse

import tensorflow as tf
from trainer import input_functions
from trainer import model
from workflow.dags.tasks.preprocess import data_pipeline_utils as utils
from workflow.dags.tasks.preprocess import features_config


def train_and_evaluate(hparams):
  """Trains and evaluates the model.

  Args:
    hparams: A TensorFlow `HParams` object to provide hyperparameters to the
      model.
  """
  train_input_fn = input_functions.get_input_fn(
      filename_list=hparams.train_data,
      tf_transform_dir=hparams.transform_dir,
      target_feature=features_config.TARGET_FEATURE,
      feature_id=features_config.ID_FEATURE,
      num_epochs=hparams.num_epochs,
      batch_size=hparams.train_batch_size)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=hparams.train_steps)

  eval_input_fn = input_functions.get_input_fn(
      filename_list=hparams.eval_data,
      tf_transform_dir=hparams.transform_dir,
      target_feature=features_config.TARGET_FEATURE,
      feature_id=features_config.ID_FEATURE,
      num_epochs=hparams.num_epochs,
      batch_size=hparams.eval_batch_size)
  schema_file = utils.read_schema(hparams.schema_file)
  raw_feature_spec = utils.get_raw_feature_spec(schema_file, 'PREDICT')

  serving_receiver_fn = lambda: input_functions.example_serving_receiver_fn(
      hparams.transform_dir, raw_feature_spec, features_config.TARGET_FEATURE,
      features_config.ID_FEATURE)
  exporter = tf.estimator.LatestExporter(
      name='driblet',
      serving_input_receiver_fn=serving_receiver_fn,
      exports_to_keep=1)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hparams.eval_steps,
      start_delay_secs=1,  # start evaluating after N seconds
      throttle_secs=2,  # evaluate every N seconds
      exporters=[exporter])

  estimator = model.create_estimator(hparams)
  tf.estimator.train_and_evaluate(
      estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def parse_arguments():
  """Initialize command line parser using argparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  # Input arguments
  parser.add_argument(
      '--train-data',
      help='GCS or local paths to training data',
      nargs='+',
      required=True)
  parser.add_argument(
      '--eval-data',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True)
  parser.add_argument(
      '--schema-file', help='File holding the schema for the input data')
  # Transform arguments
  parser.add_argument(
      '--transform-dir',
      help='Tf-transform directory with model from preprocessing step',
      required=True)
  # Training arguments
  parser.add_argument(
      '--job-dir',
      help='Google Cloud Storage location to write checkpoints and '
      'export models',
      type=str,
      required=True)
  parser.add_argument(
      '--train-steps',
      default=1000,
      help='Count of steps to run the training job for',
      required=True,
      type=int)
  parser.add_argument(
      '--train-batch-size', default=100, help='Train batch size.', type=int)
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evaluation for at each checkpoint',
      default=100,
      type=int)
  parser.add_argument(
      '--eval-batch-size', default=50, help='Eval batch size.', type=int)
  parser.add_argument(
      '--num-epochs', default=1, help='Number of epochs.', type=int)
  parser.add_argument(
      '--first-layer-size',
      default=100,
      help='Size of the first layer.',
      type=int)
  parser.add_argument(
      '--num-layers', default=2, help='Number of layers.', type=int)
  parser.add_argument(
      '--save-checkpoints-steps',
      help='Save checkpoints every this many steps.',
      default=100,
      type=int)
  parser.add_argument(
      '--keep-checkpoint-max',
      help='The maximum number of recent checkpoint files to keep.',
      default=3,
      type=int)
  parser.add_argument(
      '--dnn-optimizer',
      help='Optimizer for DNN model.',
      default='Adam',
      type=str)
  parser.add_argument(
      '--dnn-dropout', help='Dropout value (float) for DNN.', type=float)
  parser.add_argument(
      '--linear-optimizer',
      help='Optimizer for linear model.',
      default='Ftrl',
      type=str)
  # Logging arguments
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')

  return parser.parse_args()


def main():
  args = parse_arguments()
  tf.logging.set_verbosity(args.verbosity)
  hparams = tf.contrib.training.HParams(**args.__dict__)
  train_and_evaluate(hparams)


if __name__ == '__main__':
  main()
