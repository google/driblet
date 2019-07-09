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
"""Tests for dag."""

import datetime
import unittest

from airflow import models
from airflow.operators import dummy_operator
import mock
from google.cloud import bigquery
from google.cloud import storage
 import dag as dag_module


class TestDag(unittest.TestCase):

  def setUp(self):
    super(TestDag, self).setUp()
    self.test_env_variables = {
        'project_id': 'test_project',
        'bucket_name': 'test_bucket',
        'bq_dataset': 'test_dataset',
        'bq_input_table': 'test_input_table',
        'bq_output_table': 'test_output_table',
        'model_name': 'test_model',
        'model_version': 'test_version',
        'dataset_expiration': 60,
        'location': 'test_location',
        'region': 'test_region'
    }

  def test_initialize_dag(self):
    """Test if DAG correctly initialized."""
    dag = dag_module.initialize_dag()

    self.assertIsInstance(dag, models.DAG)
    self.assertListEqual(dag.tasks, [])

  @mock.patch.object(dag_module, 'configuration')
  @mock.patch.object(bigquery, 'Client')
  @mock.patch.object(storage, 'Client')
  def test_dag_has_correct_tasks(self, unused_gcs_mock, unused_bq_mock,
                                 mock_configuration):
    """Test if module has tasks."""
    mock_configuration.get.return_value = 'test_path'
    # Create dummy tasks for expected DAG
    expected_dag = models.DAG(
        dag_id='expected_dag',
        schedule_interval='0 12 * * *',
        start_date=datetime.datetime(2018, 1, 8))
    expected_task_ids = [
        'bq-to-tfrecord', 'make-predictions', 'gcs-to-bigquery',
        'gcs-delete-blob'
    ]
    for task_id in expected_task_ids:
      dummy_operator.DummyOperator(task_id=task_id, dag=expected_dag)

    actual_dag = dag_module.create_dag(self.test_env_variables)

    self.assertEqual(actual_dag.task_count, expected_dag.task_count)
    self.assertListEqual(sorted(actual_dag.task_ids), sorted(expected_task_ids))

  @mock.patch.object(dag_module, 'models', autospec=True)
  def test_get_env_variables(self, mock_models):
    """Test if returns correct environment variables."""
    mock_models.Variable.get.side_effect = self.test_env_variables.get

    actual_env_variables = dag_module.get_env_variables()

    self.assertEqual(mock_models.Variable.get.call_count,
                     len(self.test_env_variables))
    self.assertDictEqual(actual_env_variables, self.test_env_variables)


if __name__ == '__main__':
  unittest.main()
