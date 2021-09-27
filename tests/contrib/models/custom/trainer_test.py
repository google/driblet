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

import argparse
import glob
import os
import shutil
from typing import List, Text
import parameterized
import tensorflow as tf

from driblet.contrib.models.custom import trainer

# Common keys used for testing.
ID_COL = 'id_col'
PROBABILITY_OUTPUT_KEY = 'probability'
PREDICTION_OUTPUT_KEY = 'prediction'

FILE_SUFFIX = '00000-of-00001.tfrecord'
FORWARD_FEATURE_KEYS = [ID_COL, 'num_col3']
TEMP_DIR = tf.compat.v1.test.get_temp_dir()


def _get_model_export_dir(model_name: Text) -> Text:
  """Returns saved model path provided model name.

  Args:
    model_name: Name of the trained model.

  Returns:
    Path to trained model.
  """
  return glob.glob(os.path.join(TEMP_DIR, 'export', model_name, '*'))[0]


def _get_model_output_keys(hparams, estimator_type) -> List[Text]:
  """Obtains output keys of the trained model.

  Args:
    hparams: An instance of HParams object describing the hyper-parameters for
      the model.
    estimator_type: Type of the estimator. Should be one of {Regressor,
      CombinedRegressor, Classifier, CombinedClassifier}.

  Returns:
    Sorted output keys of the trained model.
  """
  hparams.model_name = 'test_model_%s' % estimator_type
  hparams.estimator_type = estimator_type
  # FLAGS.estimator_type = estimator_type
  trainer.train_and_evaluate(hparams)
  export_dir = _get_model_export_dir(hparams.model_name)

  #  tf.compat.v1.reset_default_graph()
  #  with tf.Session() as sess:
  meta_graph_def = tf.saved_model.load(export_dir, ['serve'])
  sig_def = meta_graph_def.signatures['serving_default']
  return sorted(sig_def.structured_outputs.keys())


class TrainerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Configure training hyperparameters
    transform_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'test_data')
    self.hparams = argparse.Namespace(
        train_data='{}/{}-{}'.format(transform_dir, 'train', FILE_SUFFIX),
        eval_data='{}/{}-{}'.format(transform_dir, 'eval', FILE_SUFFIX),
        transform_dir=transform_dir,
        model_name='test_model',
        job_dir=TEMP_DIR,
        save_checkpoints_steps=1,
        keep_checkpoint_max=1,
        num_layers=2,
        num_epochs=1,
        train_batch_size=10,
        eval_batch_size=10,
        eval_steps=10,
        exports_to_keep=1,
        start_delay_secs=1,
        throttle_secs=1,
        train_steps=10,
        dnn_dropout=0.0,
        learning_rate=0.001,
        first_layer_size=10,
        estimator_type='Classifier',
        linear_optimizer='Ftrl',
        dnn_optimizer='Adam',
        schema_file=os.path.join(transform_dir, 'schema.pbtxt'),
        features_config_file=os.path.join(transform_dir, 'features_config.cfg'),
        include_prediction_class=True,
        probability_output_key=PROBABILITY_OUTPUT_KEY,
        prediction_output_key=PREDICTION_OUTPUT_KEY)

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(TEMP_DIR)

  def test_train_and_evaluate_generates_checkpoints_and_saved_model(self):
    self.hparams.model_name = 'test_model_classifier'
    self.hparams.estimator_type = 'Classifier'
    trainer.train_and_evaluate(self.hparams)
    export_dir = _get_model_export_dir(self.hparams.model_name)
    saved_model = os.path.join(export_dir, 'saved_model.pb')

    # Assert that latest checkpoint exists
    self.assertTrue(tf.train.latest_checkpoint(self.hparams.job_dir))
    # Assert that saved model exists
    self.assertTrue(os.path.isfile(saved_model))

  @parameterized.parameterized.expand([
      ('Classifier'),
      ('CombinedClassifier'),
  ])
  def test_train_and_evaluate_creates_correct_classifier_signature_def(
      self, estimator_type):
    actual_model_output_keys = _get_model_output_keys(self.hparams,
                                                      estimator_type)

    # DNNClassifier and DNNLinearCombinedClassifier provides only following
    # values in prediction output:
    # ['class_ids', 'classes', 'logistic', 'logits', 'probabilities'].
    # Following are custom prediction output keys redefined in custom_model_fn
    # for classification models. Expecting these keys in signature definition.
    expected_output_keys = FORWARD_FEATURE_KEYS + [
        PREDICTION_OUTPUT_KEY, PROBABILITY_OUTPUT_KEY
    ]
    self.assertListEqual(actual_model_output_keys, expected_output_keys)

  @parameterized.parameterized.expand([
      ('Regressor'),
      ('CombinedRegressor'),
  ])
  def test_train_and_evaluate_creates_correct_regressor_signature_def(
      self, estimator_type):
    actual_model_output_keys = _get_model_output_keys(self.hparams,
                                                      estimator_type)

    # DNNRegressor and DNNLinearCombinedRegressor provides only following
    # value in prediction output: ['predictions'].
    # Following are custom prediction output keys redefined in custom_model_fn
    # for regression models. Expecting these keys in signature definition.
    expected_output_keys = FORWARD_FEATURE_KEYS + ['prediction']
    self.assertListEqual(actual_model_output_keys, expected_output_keys)


if __name__ == '__main__':
  tf.test.main()
