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

"""Tests for driblet.contrib.dags.subdags.model_trainer."""

import json
import unittest

import airflow
from airflow import models
import mock
from driblet.contrib.dags.subdags import model_trainer
from driblet.dags import constants

BASE_CONFIG = {'schedule_interval': '@once'}
GCP_CONFIG = {
    'project_id': 'test_project',
    'region': 'us-central1',
    'service_account_email': 'test@project.iam.gserviceaccount.com',
    'storage_bucket': 'gs://test_bucket'
}
TRANSFORM_CONFIG = {
    'transform_dir': 'transform_model',
    'output_dir': 'transformed_data',
    'features_config': 'features_config.cfg'
}
TRAINING_CONFIG = {
    'estimator_type': 'CombinedClassifier',
    'eval_batch_size': '100',
    'eval_steps': '100',
    'first_layer_size': '10',
    'job_dir': 'training_jobs',
    'model_name': 'mymodel',
    'num_epochs': '2',
    'num_layers': '3',
    'regenerate_data': 'True',
    'train_batch_size': '200',
    'train_steps': '1000'
}


class TestFeatureTransformDag(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_airflow_configuration = mock.patch.object(
        airflow, 'configuration', autospec=True).start()
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()

  def test_create_dag(self):
    """Tests that returned DAG contains correct DAG and tasks."""
    airflow_variables = {
        constants.BASE_CONFIG: json.dumps(BASE_CONFIG),
        constants.GCP_CONFIG: json.dumps(GCP_CONFIG),
        constants.TRAINING_CONFIG: json.dumps(TRAINING_CONFIG),
        constants.TRANSFORM_CONFIG: json.dumps(TRANSFORM_CONFIG)
    }
    self.mock_airflow_configuration.get.return_value = '.'
    self.mock_airflow_variable.get.side_effect = lambda x: airflow_variables[x]

    dag = model_trainer.create_dag(parent_dag_id='test-dag')

    self.assertIsInstance(dag, airflow.models.DAG)
    self.assertEqual(dag.task_count, 1)
    self.assertEqual(dag.tasks[0].task_id, 'training')


if __name__ == '__main__':
  unittest.main()
