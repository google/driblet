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
"""Input functions for tf.estimator.DNNLinearCombinedClassifier."""

import tensorflow as tf
import tensorflow_transform as tft
from workflow.dags.tasks.preprocess import data_pipeline_utils as utils


def convert_sparse_to_dense(sparse_tensor):
  """Converts sparse tensor to dense.

  Args:
    sparse_tensor: tf.SparseTensor object.

  Returns:
    Dense tensor.
  """
  return tf.sparse_to_dense(
      sparse_tensor.indices, [sparse_tensor.dense_shape[0], 1],
      sparse_tensor.values,
      default_value=0)


def create_feature_columns(categorical_features, numeric_features, num_buckets):
  """Creates TensorFlow feature columns for input data.

  Args:
    categorical_features: List of categorical features' keys.
    numeric_features: List of numeric features' keys.
    num_buckets: Number of buckets for vocabulary.

  Returns:
    A list of linear and dense features with transformed keys.
  """
  cat_transformed_keys = utils.get_transformed_keys(categorical_features)
  num_transformed_keys = utils.get_transformed_keys(numeric_features)
  linear_features = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=num_buckets, default_value=0)
      for key in cat_transformed_keys
  ]
  dense_features = [
      tf.feature_column.numeric_column(key, shape=())
      for key in num_transformed_keys
  ]
  return linear_features, dense_features


def get_input_fn(filename_list,
                 tf_transform_dir,
                 target_feature,
                 feature_id,
                 num_epochs,
                 batch_size,
                 shuffle=True):
  """Generates features and labels for training and evaluation.

  Args:
    filename_list: List of gzipped TFRecord files to read the data from.
    tf_transform_dir: Directory in which the tf.Transform model was written.
    target_feature: Key for target feature.
    feature_id: Key for id field in input data.
    num_epochs: Number of epochs to read the data.
    batch_size: An integer representing the number of batch size while reading
      the data.
    shuffle: Boolean value if set True shuffles the input data.

  Returns:
    An input function that can be passed to tf.Estimator.
  """

  def input_fn():
    """Supplies input data to the model.

    Returns:
      features: A dictionary of input features.
      target: A Tensor of target feature.
    """
    tf_transform_output = tft.TFTransformOutput(tf_transform_dir)
    feature_spec = tf_transform_output.transformed_feature_spec()
    dataset = tf.data.TFRecordDataset(filename_list)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: tf.parse_example(x, feature_spec))
    features = dataset.make_one_shot_iterator().get_next()
    target = features.pop(target_feature)
    # Convert id feature to dense tensor to be used in
    # tf.estimator.forward_features() during inference time.
    features[feature_id] = convert_sparse_to_dense(features[feature_id])
    return features, target

  return input_fn


def example_serving_receiver_fn(tf_transform_dir, raw_feature_spec,
                                target_feature, feature_id):
  """Creates serving function that is used during inference.

  Args:
    tf_transform_dir: A directory in which the tf.Transform model was written
      during the preprocessing step.
    raw_feature_spec: A dictionary of raw feature spec for input data.
    target_feature: Key for target feature.
    feature_id: Key for id field in input data.

  Returns:
    An instance of tf.estimator.export.ServingInputReceiver that parses input
    data by applying transformation from saved tf.Transform graph.
  """
  if target_feature in raw_feature_spec:
    raw_feature_spec.pop(target_feature)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()
  features = serving_input_receiver.features
  transform_output = tft.TFTransformOutput(tf_transform_dir)
  transformed_features = transform_output.transform_raw_features(features)
  transformed_features[feature_id] = (
      convert_sparse_to_dense(transformed_features[feature_id]))
  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)
