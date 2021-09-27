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

"""Tests for driblet.contrib.dags.subdags.storage_cleaner."""

import json
import unittest

from airflow import models
from google.cloud import storage
import mock
from driblet.contrib.dags.subdags import storage_cleaner
from driblet.dags import constants

BASE_CONFIG = {'schedule_interval': '@once'}
GCP_CONFIG = {
    'project_id': 'test_project',
    'region': 'us-central1',
    'service_account_email': 'test@project.iam.gserviceaccount.com',
    'storage_bucket': 'gs://test_bucket'
}


class StorageCleanerDagTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()
    self.mock_gcs_client = mock.patch.object(
        storage, 'Client', autospec=True).start()

  def test_create_dag(self):
    """Tests that returned DAG contains correct DAG and tasks."""
    airflow_variables = {
        constants.BASE_CONFIG: json.dumps(BASE_CONFIG),
        constants.GCP_CONFIG: json.dumps(GCP_CONFIG)
    }
    self.mock_airflow_variable.get.side_effect = lambda x: airflow_variables[x]
    prefixes = ['test_folder1', 'test_folder2']

    dag = storage_cleaner.create_dag(
        parent_dag_id='test-dag-id', prefixes=prefixes)
    task = dag.tasks[0]

    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(dag.task_count, 1)
    self.assertEqual(task._gcs_bucket, 'test_bucket')
    self.assertListEqual(task._prefixes, prefixes)


if __name__ == '__main__':
  unittest.main()
