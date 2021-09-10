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

"""Tests for driblet.dags.subdags.dataset_split."""

import json
import unittest

import airflow
from airflow import models
import mock
import parameterized

from driblet.dags import constants
from driblet.dags.subdags import dataset_split

BASE_CONFIG = {'schedule_interval': '@once'}
DATASET_SPLIT_CONFIG = {
    'id_column': 'test_id',
    'input_table': 'project.dataset.features',
    'train_dest_table': 'project.dataset.train',
    'eval_dest_table': 'project.dataset.eval',
    'test_dest_table': 'project.dataset.test',
    'train_proportion': '80',
    'eval_proportion': '10',
    'test_proportion': '10'
}


class TestBatchPredictionDag(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_airflow_configuration = mock.patch.object(
        airflow, 'configuration', autospec=True).start()
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()

  @parameterized.parameterized.expand([('80', '10', '10'), ('65', '20', '15')])
  def test_create_dataset_split_proportions_creates_correct_proportions(
      self, train_proportion, eval_proportion, test_proportion):
    test_config = DATASET_SPLIT_CONFIG.copy()
    test_config['train_proportion'] = train_proportion
    test_config['eval_proportion'] = eval_proportion
    test_config['test_proportion'] = test_proportion

    split_proportions = dataset_split.create_dataset_split_proportions(
        test_config)
    expected_split_proportions = {
        'train': {
            'dest_table': 'project.dataset.train',
            'lower_bound': 0,
            'upper_bound': int(train_proportion)
        },
        'eval': {
            'dest_table': 'project.dataset.eval',
            'lower_bound': int(train_proportion),
            'upper_bound': sum([int(train_proportion),
                                int(eval_proportion)])
        },
        'test': {
            'dest_table': 'project.dataset.test',
            'lower_bound': sum([int(train_proportion),
                                int(eval_proportion)]),
            'upper_bound': 100
        }
    }
    self.assertDictEqual(split_proportions, expected_split_proportions)

  def test_create_dataset_split_proportions_raises_asserion_error(self):
    test_config = DATASET_SPLIT_CONFIG.copy()
    test_config['train_proportion'] = '80'
    test_config['eval_proportion'] = '20'
    test_config['test_proportion'] = '10'

    with self.assertRaises(AssertionError):
      dataset_split.create_dataset_split_proportions(test_config)

  def test_create_dag(self):
    """Tests that returned DAG contains correct DAG and tasks."""
    airflow_variables = {
        constants.BASE_CONFIG: json.dumps(BASE_CONFIG),
        constants.DATASET_SPLIT_CONFIG: json.dumps(DATASET_SPLIT_CONFIG),
        constants.TRAINING_SUFFIX: 'training'
    }
    test_sql = (
        'SELECT * FROM `{bq_input_table}` WHERE id_column = "{id_column}" AND '
        'lower_bound = {proportion_lower_bound} AND upper_bound = '
        '{proportion_upper_bound};')
    self.mock_airflow_configuration.get.return_value = '.'
    self.mock_airflow_variable.get.side_effect = (
        lambda x: airflow_variables[x])
    self.mock_open = mock.mock_open(read_data=test_sql)
    # split_proportions indicate what proportion is allocated for each of three
    # datasets: train, eval and test
    split_proportions = {
        'train': {
            'lower_bound': 0,
            'upper_bound': 80
        },
        'eval': {
            'lower_bound': 80,
            'upper_bound': 90
        },
        'test': {
            'lower_bound': 90,
            'upper_bound': 100
        }
    }
    expected_task_ids = [f'dataset-split-{key}' for key in split_proportions]
    expected_dest_tables = [
        f'project.dataset.{table_name}' for table_name in split_proportions
    ]

    with mock.patch('builtins.open', self.mock_open, create=True):
      dag = dataset_split.create_dag(parent_dag_id='test-dag-id')

    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(dag.task_count, 3)
    # Loop through the dag tasks and test its parameters.
    for (task, expected_task_id, expected_dest_table,
         proportion) in zip(dag.tasks, expected_task_ids, expected_dest_tables,
                            split_proportions.values()):
      expected_sql = test_sql.format(
          bq_input_table='project.dataset.features_training',
          id_column='test_id',
          proportion_lower_bound=proportion['lower_bound'],
          proportion_upper_bound=proportion['upper_bound'])
      self.assertEqual(task.task_id, expected_task_id)
      self.assertEqual(task.sql, expected_sql)
      self.assertEqual(task.destination_dataset_table, expected_dest_table)
      self.assertEqual(task.write_disposition, 'WRITE_TRUNCATE')
      self.assertFalse(task.use_legacy_sql)


if __name__ == '__main__':
  unittest.main()
