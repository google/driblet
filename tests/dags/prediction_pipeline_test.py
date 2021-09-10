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
"""Tests for driblet.dags.subdags.prediction_pipeline."""

import json
import unittest

import airflow
from airflow import models
import mock

from driblet.dags import constants
from driblet.dags import prediction_pipeline
from driblet.dags import utils
from driblet.dags.subdags import bq_query
from driblet.dags.subdags import mlwp_features

_BASE_CONFIG = {'schedule_interval': '@once', 'pipeline_mode': 'prediction'}
_PREDICTION_PIPELINE_CONFIG = {
    'feature_generator': 'mlwp',
    'features_sql_params': {
        'raw_input_table': 'project.dataset.raw',
        'features_output_table': 'project.dataset.features',
    },
    'prediction_sql_params': {
        'prediction_output_table': 'project.dataset.predictions',
        'model_name': 'project.dataset.model',
        'features_input_table': 'project.dataset.features'
    }
}


class PredictionPipelineTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_airflow_configuration = mock.patch.object(
        airflow, 'configuration', autospec=True).start()
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()
    self.mock_mwp_features = mock.patch.object(
        mlwp_features, 'create_operator', autospec=True).start()
    self.mock_bq_query = mock.patch.object(
        bq_query, 'create_operator', autospec=True).start()

  def test_create_dag(self):
    """Tests that returned DAG contains correct DAG with tasks."""
    self.mock_airflow_configuration.get.return_value = '.'
    airflow_variables = {
        constants.BASE_CONFIG:
            json.dumps(_BASE_CONFIG),
        constants.PREDICTION_PIPELINE_CONFIG:
            json.dumps(_PREDICTION_PIPELINE_CONFIG)
    }
    self.mock_airflow_variable.get.side_effect = (
        lambda x: airflow_variables[x])

    dag = prediction_pipeline.create_dag()

    self.mock_mwp_features.side_effect = [
        utils.create_test_operator(dag=dag, task_id='test_mlwp')
    ]
    self.mock_bq_query.side_effect = [
        utils.create_test_operator(dag=dag, task_id='test_query')
    ]
    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(len(dag.tasks), 2)


if __name__ == '__main__':
  unittest.main()
