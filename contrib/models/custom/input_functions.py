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

"""Input functions for TensorFlow Estimator."""

from typing import Any, Callable, Dict, List, Text, Tuple
import tensorflow as tf
import tensorflow_transform as tft
from driblet.contrib.models.custom.feature_transformer import utils

_SHUFFLE_BUFFER_SIZE = 1024


def convert_sparse_to_dense(sparse_tensor: tf.SparseTensor) -> tf.Tensor:
  """Converts sparse tensor to dense.

  Args:
    sparse_tensor: tf.SparseTensor object.

  Returns:
    Dense tensor.
  """
  default_value = '' if sparse_tensor.dtype == tf.string else 0
  return tf.sparse.to_dense(sparse_tensor, default_value=default_value)


def create_feature_columns(
    categorical_features: List[Text],
    numeric_features: List[Text],
    num_buckets: int,
    estimator_type: Text,
) -> Tuple[List[Any], List[Any]]:
  """Creates TensorFlow feature columns for input data.

  Args:
    categorical_features: Categorical features' keys.
    numeric_features: Numeric features' keys.
    num_buckets: Number of buckets for vocabulary.
    estimator_type: Type of the estimator. Should be one of {Regressor,
      CombinedRegressor, Classifier, CombinedClassifier}.

  Returns:
    A lists of linear and dense features with transformed keys.
  """
  cat_transformed_keys = utils.get_transformed_keys(categorical_features)
  num_transformed_keys = utils.get_transformed_keys(numeric_features)
  # Embedding dimension is generated as suggested in go/text-recipes.
  embedding_dimension = int(6 * num_buckets**0.25)

  dense_features = [
      tf.feature_column.numeric_column(key, shape=())
      for key in num_transformed_keys
  ]
  # TODO(zmtbnv): Move hardcoded estimator_type definitions to common enum
  # that can be used both by input_functions and models.
  linear_features = []
  for key in cat_transformed_keys:
    if estimator_type in ['Regressor', 'Classifier']:
      dense_features.append(
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_identity(
                  key, num_buckets=num_buckets, default_value=0),
              dimension=embedding_dimension))
    elif estimator_type in ['CombinedRegressor', 'CombinedClassifier']:
      linear_features.append(
          tf.feature_column.categorical_column_with_identity(
              key, num_buckets=num_buckets, default_value=0))
  return linear_features, dense_features


def get_input_fn(filename_patterns: List[Any],
                 tf_transform_dir: Text,
                 target_feature: Text,
                 forward_features: List[Text],
                 num_epochs: int,
                 batch_size: int,
                 shuffle: bool = True) -> Callable[[], Tuple[Any, Any]]:
  """Returns an input function that for Tensorflow Estimator.

  Args:
    filename_patterns: List of TFRecord files to read the data from.
    tf_transform_dir: Directory in which the tf.Transform model was written.
    target_feature: Key for target feature.
    forward_features: Features to forward during prediction time.
    num_epochs: Number of epochs to read the data.
    batch_size: Number of batch size while reading the data.
    shuffle: Shuffles the input data if set True.

  Returns:
    An input function that can be passed to tf.Estimator.
  """

  def input_fn() -> Tuple[Any, Any]:
    """Supplies features and labels for training and evaluation of the model.

    Returns:
      features: Input features.
      target: Target feature.
    """
    tf_transform_output = tft.TFTransformOutput(tf_transform_dir)
    feature_spec = tf_transform_output.transformed_feature_spec()
    try:
      files_list = tf.io.gfile.glob(filename_patterns)
    except tf.compat.v1.OpError as error:
      raise ValueError(
          'Files do not exist in provided file pattern {}. {}'.format(
              filename_patterns, error))
    dataset = tf.data.TFRecordDataset(files_list)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: tf.io.parse_example(x, feature_spec))
    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    target = features.pop(target_feature)
    # Convert sparse features in forward_features to dense tensor to be used in
    # tf.estimator.forward_features() during inference time.
    for feature in forward_features:
      features[feature] = convert_sparse_to_dense(features[feature])
    return features, target

  return input_fn


def example_serving_receiver_fn(
    tf_transform_dir: Text, raw_feature_spec: Dict[Text,
                                                   Any], target_feature: Text,
    forward_features: List[Text]) -> tf.estimator.export.ServingInputReceiver:
  """Creates serving function used during inference.

  Args:
    tf_transform_dir: A directory in which the tf.Transform model was written
      during the preprocessing step.
    raw_feature_spec: A dictionary of raw feature spec for input data.
    target_feature: Key for target feature.
    forward_features: Features to forward during prediction time.

  Returns:
    An instance of tf.estimator.export.ServingInputReceiver that parses input
    data by applying transformation from saved tf.Transform graph.
  """
  feature_spec = raw_feature_spec.copy()
  if target_feature in feature_spec:
    feature_spec.pop(target_feature)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()
  transform_output = tft.TFTransformOutput(tf_transform_dir)
  transformed_features = transform_output.transform_raw_features(
      serving_input_receiver.features)
  for feature in forward_features:
    transformed_features[feature] = (
        convert_sparse_to_dense(transformed_features[feature]))
  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)
