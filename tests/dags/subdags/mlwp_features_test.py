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
"""Tests for driblet.dags.subdags.mlwp_features."""

import json
import unittest

from airflow import models
from airflow.operators import python_operator
import mock

import parameterized
from driblet.dags import constants
from driblet.dags.subdags import mlwp_features

BASE_CONFIG = {
    'schedule_interval': '@once',
    'feature_generator': 'custom',
    'pipeline_mode': 'train'
}
GCP_CONFIG = {'project_id': 'test_project'}
CUSTOM_QUERY_CONFIG = {
    'params': {
        'input_table': 'project.dataset.input_table',
        'output_table': 'project.dataset.output_table'
    }
}
MWP_CONFIG = {
    'dataset_id':
        'mlwp_data',
    'run_id':
        '01',
    'analytics_table':
        'bigquery-public-data.google_analytics_sample.ga_sessions_*',
    'slide_interval_in_days':
        7,
    'lookback_window_size_in_days':
        30,
    'lookback_window_gap_in_days':
        1,
    'prediction_window_gap_in_days':
        1,
    'prediction_window_size_in_days':
        14,
    'sum_values':
        'totals_visits;totals_hits',
    'avg_values':
        'totals_visits;totals_hits',
    'count_values':
        'channelGrouping:[Organic Search,Social,Direct,Referral,Paid '
        'Search,Affiliates]:[Other]',
    'mode_values':
        'hits_eCommerceAction_action_type:[3]:[Others]',
    'proportions_values':
        'channelGrouping:[Organic Search,Social,Direct,Referral,Paid '
        'Search,Affiliates]:[Others]',
    'latest_values':
        'device_isMobile:[false,true]:[Others]'
}
SQL_SCRIPT = 'SELECT * FROM source_table;'


class MlwpFeaturesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()
    airflow_variables = {
        constants.BASE_CONFIG: json.dumps(BASE_CONFIG),
        constants.GCP_CONFIG: json.dumps(GCP_CONFIG),
        constants.MWP_CONFIG: json.dumps(MWP_CONFIG)
    }
    self.mock_airflow_variable.get.side_effect = lambda x: airflow_variables[x]

  @parameterized.parameterized.expand([
      (constants.PipelineMode.TRAIN, mlwp_features.TRAIN_TASK_ID),
      (constants.PipelineMode.PREDICT, mlwp_features.PREDICTION_TASK_ID)
  ])
  def test_create_operator_returns_python_operator(self, pipeline_mode,
                                                   task_id):
    expected_task_id = task_id

    operator = mlwp_features.create_operator(
        parent_dag_id='test-dag-id',
        pipeline_mode=pipeline_mode)

    self.assertIsInstance(operator, python_operator.PythonOperator)
    self.assertEqual(operator.task_id, expected_task_id)

  def test_create_dag_raises_value_error_on_incorrect_pipeline_mode(self):
    incorrect_pipeline_mode = 'test'

    with self.assertRaises(ValueError):
      mlwp_features.create_operator(
          parent_dag_id='test-dag-id', pipeline_mode=incorrect_pipeline_mode)


if __name__ == '__main__':
  unittest.main()
