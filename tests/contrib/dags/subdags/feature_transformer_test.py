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

"""Tests for driblet.contrib.dags.subdags.feature_transformer."""

import datetime
import json
import os
import unittest

import airflow
from airflow import models
import mock
import parameterized
from driblet.contrib.dags.subdags import feature_transformer
from driblet.dags import constants

BASE_CONFIG = {'schedule_interval': '@once'}
GCP_CONFIG = {
    'project_id': 'test_project',
    'region': 'us-central1',
    'service_account_email': 'test@project.iam.gserviceaccount.com',
    'storage_bucket': 'gs://test_bucket'
}
TRANSFORM_CONFIG = {
    'features_config': 'features_config.cfg',
    'output_dir': 'transformed_data',
    'transform_dir': 'transform_model'
}
DATASET_SPLIT_CONFIG = {
    'id_column': 'test_id_column',
    'input_table': 'project.dataset.features',
    'eval_dest_table': 'project.dataset.eval',
    'test_dest_table': 'project.dataset.test',
    'train_dest_table': 'project.dataset.train'
}


class TestFeatureTransformDag(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)

    self.mock_airflow_configuration = mock.patch.object(
        airflow, 'configuration', autospec=True).start()
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()

  @parameterized.parameterized.expand([[True], [False]])
  def test_create_dag(self, train_mode):
    """Tests that returned DAG contains correct DAG and tasks."""
    BASE_CONFIG['train_mode'] = train_mode
    airflow_variables = {
        constants.BASE_CONFIG: json.dumps(BASE_CONFIG),
        constants.GCP_CONFIG: json.dumps(GCP_CONFIG),
        constants.DATASET_SPLIT_CONFIG: json.dumps(DATASET_SPLIT_CONFIG),
        constants.TRANSFORM_CONFIG: json.dumps(TRANSFORM_CONFIG)
    }
    self.mock_airflow_variable.get.side_effect = (
        lambda x: airflow_variables[x])
    self.mock_airflow_configuration.get.return_value = '.'
    expected_task_id = 'feature_transform'
    expected_py_file = 'feature_transformer.py'
    expected_options = {
        'labels': {
            'airflow-version': 'v1-10-15'
        },
        'project_id': 'test_project',
        'data_source': 'bigquery',
        'mode_train': str(train_mode),
        'transform_dir': 'gs://test_bucket/transform_model',
        'features_config': 'gs://test_bucket/features_config.cfg',
    }
    if train_mode:
      expected_options.update({
          'all_data': 'project.dataset.features_training',
          'train_data': 'project.dataset.train',
          'eval_data': 'project.dataset.eval',
          'predict_data': 'project.dataset.test',
          'output_dir': 'gs://test_bucket/transformed_data/training/'
      })
    else:
      table_suffix = datetime.datetime.now().strftime('%Y%m%d')
      expected_options.update({
          'predict_data': f'project.dataset.features_prediction_{table_suffix}',
          'output_dir': 'gs://test_bucket/transformed_data/prediction/'
      })
    expected_dataflow_options = {'project': 'test_project'}

    dag = feature_transformer.create_dag(
        parent_dag_id='test_dag', train_mode=train_mode)
    task = dag.tasks[0]

    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(dag.task_count, 1)
    self.assertEqual(task.task_id, expected_task_id)
    self.assertEqual(os.path.basename(task.py_file), expected_py_file)
    self.assertDictEqual(task.options, expected_options)
    self.assertDictEqual(task.dataflow_default_options,
                         expected_dataflow_options)


if __name__ == '__main__':
  unittest.main()
