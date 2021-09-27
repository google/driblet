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
"""Tests for driblet.contrib.dags.subdags.training_pipeline."""

import json
import unittest

from airflow import models
from airflow.operators import subdag_operator
from airflow.utils import db
import mock

from driblet.contrib.dags import training_pipeline
from driblet.contrib.dags.subdags import model_trainer
from driblet.contrib.dags.subdags import storage_cleaner
from driblet.dags import constants
from driblet.dags import utils

TRAINING_CONFIG = {
    'estimator_type': 'CombinedClassifier',
    'eval_batch_size': '100',
    'eval_steps': '100',
    'first_layer_size': '10',
    'job_dir': 'training_jobs',
    'model_name': 'mymodel',
    'num_epochs': '2',
    'num_layers': '3',
    'regenerate_data': 'True',
    'train_batch_size': '200',
    'train_steps': '1000'
}


class PredictionPipelineDagTest(unittest.TestCase):

  def setUp(self):
    super(PredictionPipelineDagTest, self).setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()

    self.parent_dag_id = training_pipeline._DAG_ID
    self.mock_storage_cleaner = mock.patch.object(
        storage_cleaner, 'create_dag', autospec=True).start()
    self.mock_training = mock.patch.object(
        model_trainer, 'create_dag', autospec=True).start()

    db.initdb([])

  def test_create_dag(self):
    """Tests that returned DAG contains correct DAG, subdag and tasks."""
    airflow_test_variables = {
        constants.TRAINING_CONFIG: json.dumps(TRAINING_CONFIG)
    }
    self.mock_variable.get.side_effect = lambda x: airflow_test_variables[x]
    self.mock_storage_cleaner.side_effect = [
        utils.create_test_dag(
            dag_id=f'{self.parent_dag_id}.{storage_cleaner.SUB_DAG_ID}')
    ]
    self.mock_training.side_effect = [
        utils.create_test_dag(
            dag_id=f'{self.parent_dag_id}.{model_trainer.SUB_DAG_ID}')
    ]
    expected_task_ids = [storage_cleaner.SUB_DAG_ID, model_trainer.SUB_DAG_ID]

    dag = training_pipeline.create_dag()

    self.assertIsInstance(dag, models.DAG)
    self.assertEqual(dag.task_count, len(expected_task_ids))
    for task in dag.tasks:
      self.assertIsInstance(task, subdag_operator.SubDagOperator)
    for idx, task_id in enumerate(expected_task_ids):
      self.assertEqual(dag.tasks[idx].task_id, task_id)


if __name__ == '__main__':
  unittest.main()
