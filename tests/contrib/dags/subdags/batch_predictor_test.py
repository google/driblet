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

"""Tests for driblet.contrib.dags.subdags.batch_predictor."""

import datetime
import json
import unittest

import airflow
from airflow import models
import mock
import pytz
from driblet.contrib.dags.subdags import batch_predictor
from driblet.dags import constants

BASE_CONFIG = {'schedule_interval': '@once', 'local_timezone': 'Asia/Tokyo'}
GCP_CONFIG = {
    'project_id': 'test_project',
    'region': 'us-central1',
    'service_account_email': 'test@project.iam.gserviceaccount.com',
    'storage_bucket': 'gs://test_bucket'
}
TRANSFORM_CONFIG = {'output_dir': 'transformed_data'}
PREDICTION_CONFIG = {
    'model_name': 'test_model',
    'model_version': 'v0',
    'output_dir': 'prediction_results'
}


class TestBatchPredictionDag(unittest.TestCase):

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
        constants.TRANSFORM_CONFIG: json.dumps(TRANSFORM_CONFIG),
        constants.PREDICTION_CONFIG: json.dumps(PREDICTION_CONFIG)
    }
    self.mock_airflow_variable.get.side_effect = lambda x: airflow_variables[x]
    self.mock_airflow_configuration.get.return_value = '.'
    expected_input_paths = [
        'gs://test_bucket/transformed_data/prediction/predict-*'
    ]
    local_timezone = pytz.timezone('Asia/Tokyo')
    today = datetime.datetime.now(local_timezone)
    date_suffix = today.strftime('%Y%m%d_%H%M%S')
    expected_output_path = f'gs://test_bucket/prediction_results/{date_suffix}'
    expected_model_name = 'test_model'
    expected_model_version = 'v0'

    dag = batch_predictor.create_dag(parent_dag_id='test-dag-id')
    task = dag.tasks[0]

    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(dag.task_count, 1)
    self.assertEqual(task.task_id, 'batch-predictor')
    self.assertListEqual(task._input_paths, expected_input_paths)
    self.assertEqual(task._output_path, expected_output_path)
    self.assertEqual(task._model_name, expected_model_name)
    self.assertEqual(task._version_name, expected_model_version)


if __name__ == '__main__':
  unittest.main()
