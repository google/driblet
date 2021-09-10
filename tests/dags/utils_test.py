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
"""Tests for driblet.dags.test_utils."""
import datetime
import json
import unittest

from airflow import models
from airflow import utils as airflow_utils
from airflow.operators import dummy_operator
import mock

from driblet.dags import utils


class TestUtilsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()

  def test_initialize_airflow_dag_provides_correct_dag(self):
    dag = utils.create_dag('test_dag', 1, 1, 1)

    self.assertIsInstance(dag, models.DAG)
    self.assertListEqual(dag.tasks, [])

  def test_initialize_airflow_dag_updates_default_args(self):
    test_kwargs = {'key1': 1, 'key2': 'value'}
    expected_default_args = {
        'key1': 1,
        'key2': 'value',
        'retries': 1,
        'retry_delay': datetime.timedelta(0, 60),
        'start_date': airflow_utils.dates.days_ago(1)
    }

    dag = utils.create_dag('test_dag', 1, 1, 1, **test_kwargs)

    self.assertDictEqual(dag.default_args, expected_default_args)

  def test_retrieve_airflow_variable_as_dict_parses_json(self):
    test_dict = {'key1': {'key2': 'value2'}}
    self.mock_airflow_variable.get.side_effect = (
        lambda x: json.dumps(test_dict[x]))
    expected_dict = test_dict.get('key1')

    actual = utils.get_airflow_variable_as_dict('key1')

    self.assertDictEqual(actual, expected_dict)

  def test_retrieve_airflow_variable_as_dict_parses_raises_exception(self):
    test_dict = {'correct_key': 'test_value'}
    self.mock_airflow_variable.get.side_effect = (
        lambda x: json.dumps(test_dict[x]))

    with self.assertRaises(Exception):
      utils.get_airflow_variable_as_dict('wrong_key')

  def test_initialize_airflow_dag_set_local_macros(self):
    local_macros = {'sample_dataset': 'predictions_table'}
    expected_user_defined_macros = {'sample_dataset': 'predictions_table'}

    dag = utils.create_dag('test_dag', 1, 1, 1, 1, local_macros)

    self.assertDictEqual(dag.user_defined_macros, expected_user_defined_macros)

  def test_create_test_dag_returns_dag_instance(self):
    dag = utils.create_test_dag()

    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(dag.task_count, 1)

  def test_create_test_dag_returns_dummy_operator_instance(self):
    operator = utils.create_test_operator()

    self.assertIsInstance(operator, dummy_operator.DummyOperator)


if __name__ == '__main__':
  unittest.main()
