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
"""Tests for driblet.dags.subdags.bq_query."""

import json
import unittest
import airflow
from airflow import models
from airflow.contrib.operators import bigquery_operator
import mock

from driblet.dags import constants
from driblet.dags.subdags import bq_query

_BASE_CONFIG = {'schedule_interval': '@once'}
_TEST_SQL = ('CREATE OR REPLACE TABLE `{bq_output_table}` AS ( SELECT id '
             'FROM `{bq_input_table}`)')
_SQL_PARAMS = {
    'bq_input_table': 'input_table',
    'bq_output_table': 'output_table'
}


class CustomizeQueryExecutorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.airflow_variables = {constants.BASE_CONFIG: json.dumps(_BASE_CONFIG)}
    self.mock_airflow_configuration = mock.patch.object(
        airflow, 'configuration', autospec=True).start()
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()
    self.mock_airflow_configuration.get.return_value = '.'
    self.mock_airflow_variable.get.side_effect = (
        lambda x: self.airflow_variables[x])
    self.mock_open = mock.mock_open(read_data=_TEST_SQL)

  def test_create_dag(self):
    """Tests that returned DAG contains correct DAG and tasks."""
    with mock.patch('builtins.open', self.mock_open, create=True):
      operator = bq_query.create_operator(
          parent_dag_id='test_dag_id',
          sql_params=_SQL_PARAMS,
          sql_path='dummy.sql')

    self.assertIsInstance(operator, bigquery_operator.BigQueryOperator)

    expected_task_id = bq_query.TASK_ID
    expected_sql = _TEST_SQL.format(**_SQL_PARAMS)

    self.assertEqual(operator.task_id, expected_task_id)
    self.assertEqual(operator.sql, expected_sql)

  def test_create_dag_raise_key_error(self):
    """Tests that KeyError is raised if key is missing in parameter config."""
    self.mock_open = mock.mock_open(read_data=_TEST_SQL)

    sql_params = {}

    with self.assertRaises(KeyError):
      with mock.patch('builtins.open', self.mock_open, create=True):
        bq_query.create_operator(
            parent_dag_id='test_dag_id',
            sql_params=sql_params,
            sql_path='dummy.sql')


if __name__ == '__main__':
  unittest.main()
