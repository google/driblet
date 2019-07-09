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
"""Tests for trainer.task."""

import glob
import os
import shutil
import tempfile

import task
import tensorflow as tf

_TEST_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'test_data')
_TRANSFORM_DIR = '%s/transformer' % _TEST_DATA_DIR
_TRANSFORMED_DATA_DIR = '%s/transformed_data' % _TEST_DATA_DIR
_SCHEMA_FILE = '%s/schema.pbtxt' % _TRANSFORM_DIR
_MODEL_NAME = 'driblet'
_FILE_SUFFIX = '00000-of-00001.tfrecord'
_TRAIN_DATA = '{}/{}-{}'.format(_TRANSFORMED_DATA_DIR, 'train', _FILE_SUFFIX)
_EVAL_DATA = '{}/{}-{}'.format(_TRANSFORMED_DATA_DIR, 'eval', _FILE_SUFFIX)


class TaskTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TaskTest, cls).setUpClass()
    # Temporary directory to keep trained model and checkpoint files
    cls.tempdir = tempfile.mkdtemp()
    # Configure training hyperparameters
    cls.hparams = tf.contrib.training.HParams(
        train_data=_TRAIN_DATA,
        eval_data=_EVAL_DATA,
        transform_dir=_TRANSFORM_DIR,
        job_dir=cls.tempdir,
        save_checkpoints_steps=1,
        keep_checkpoint_max=1,
        num_layers=2,
        num_epochs=2,
        train_batch_size=100,
        eval_batch_size=100,
        eval_steps=100,
        train_steps=100,
        dnn_dropout=0.7,
        schema_file=_SCHEMA_FILE,
        dnn_optimizer='Adam',
        linear_optimizer='Ftrl',
        first_layer_size=10)
    # Run train_and_evaluate with test data
    task.train_and_evaluate(cls.hparams)
    # Setting exported model directory after training has been finished
    cls.export_dir = glob.glob(
        os.path.join(cls.tempdir, 'export', _MODEL_NAME, '*'))

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.tempdir)

  def test_train_and_evaluate_saved_model(self):
    """Tests if train_and_evaluate saves checkpoints and exports model."""
    saved_model = os.path.join(self.export_dir[0], 'saved_model.pb')
    # Assert that latest checkpoint exists
    self.assertTrue(tf.train.latest_checkpoint(self.tempdir))
    # Assert that saved model exists
    self.assertTrue(os.path.isfile(saved_model))

  def test_train_and_evaluate_verify_signature_def(self):
    """Tests if saved model contains correct signature output keys."""
    with self.test_session() as sess:
      meta_graph_def = tf.saved_model.loader.load(
          sess, [tf.saved_model.tag_constants.SERVING], self.export_dir[0])
      sig_def = meta_graph_def.signature_def[
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      # Following are custom prediction output keys set in custom_model_fn.
      # Expecting these keys in signature definition, instead of default
      # prediction output keys of tf.estimator.DNNLinearCombinedClassifier:
      # ['class_ids', 'classes', 'logistic', 'logits', 'probabilities']
      expected_output_keys = ['id', 'prediction', 'probability']
      self.assertListEqual(sorted(sig_def.outputs), expected_output_keys)


if __name__ == '__main__':
  tf.test.main()
