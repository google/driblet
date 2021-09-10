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
"""Tests for driblet.dags.feature_pipeline."""

import json
import unittest
import airflow

from airflow import models
from airflow.contrib.operators import bigquery_operator
from airflow.operators import python_operator
import mock

import parameterized
from driblet.dags import constants
from driblet.dags import feature_pipeline

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


class FeaturePipelineDagTest(unittest.TestCase):

  def setUp(self):
    super(FeaturePipelineDagTest, self).setUp()
    self.addCleanup(mock.patch.stopall)

    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()
    self.mock_airflow_configuration = mock.patch.object(
        airflow, 'configuration', autospec=True).start()
    self.mock_airflow_configuration.get.return_value = '.'
    self.airflow_variables = {
        constants.GCP_CONFIG: json.dumps(GCP_CONFIG),
        constants.MWP_CONFIG: json.dumps(MWP_CONFIG),
        constants.CUSTOM_QUERY_CONFIG: json.dumps(CUSTOM_QUERY_CONFIG)
    }
    self.parent_dag_id = feature_pipeline._DAG_ID

  @parameterized.parameterized.expand([
      (constants.FeatureGenerator.CUSTOM, bigquery_operator.BigQueryOperator),
      (constants.FeatureGenerator.MLWP, python_operator.PythonOperator)
  ])
  def test_create_dag_provides_correct_dag_components(self, feature_generator,
                                                      expected_operator):
    """Tests that returned DAG contains correct DAG, subdag and tasks."""
    BASE_CONFIG['feature_generator'] = feature_generator.value
    self.airflow_variables[constants.BASE_CONFIG] = json.dumps(BASE_CONFIG)
    self.mock_airflow_variable.get.side_effect = (
        lambda x: self.airflow_variables[x])

    with mock.patch('builtins.open', mock.mock_open(read_data=SQL_SCRIPT)):
      dag = feature_pipeline.create_dag()

    self.assertIsInstance(dag, models.DAG)
    self.assertIsInstance(dag.tasks[0], expected_operator)

  def test_create_dag_raises_valueerror_on_wrong_feature_generator(self):
    BASE_CONFIG['feature_generator'] = 'unknown'
    self.airflow_variables[constants.BASE_CONFIG] = json.dumps(BASE_CONFIG)
    self.mock_airflow_variable.get.side_effect = (
        lambda x: self.airflow_variables[x])

    with self.assertRaises(ValueError):
      feature_pipeline.create_dag()


if __name__ == '__main__':
  unittest.main()
