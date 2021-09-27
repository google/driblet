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

"""Tests for driblet.contrib.dags.subdags.training_dag."""

import datetime
import json
import unittest

from airflow import models
from airflow import utils
from airflow.contrib.operators import mlengine_operator
from airflow.operators import dummy_operator
from airflow.operators import python_operator
from airflow.utils import trigger_rule
import mock

from driblet.contrib.dags import model_deployer_pipeline
from driblet.dags import constants

BASE_CONFIG = {'schedule_interval': '@once'}
GCP_CONFIG = {'project_id': 'project_id'}
MODEL_DEPLOY_CONFIG = {
    'model_name': 'model_name',
    'deployment_uri': 'gs://path/to/model'
}


class ModelDeployerDagTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.schedule = 1
    self.project_id = GCP_CONFIG['project_id']
    self.model_name = MODEL_DEPLOY_CONFIG['model_name']
    self.deployment_uri = MODEL_DEPLOY_CONFIG['deployment_uri']
    airflow_variables = {
        constants.BASE_CONFIG: json.dumps(BASE_CONFIG),
        constants.GCP_CONFIG: json.dumps(GCP_CONFIG),
        constants.MODEL_DEPLOY_CONFIG: json.dumps(MODEL_DEPLOY_CONFIG)
    }
    self.mock_airflow_variable = mock.patch.object(
        models, 'Variable', autospec=True).start()
    self.mock_airflow_variable.get.side_effect = lambda x: airflow_variables[x]
    self.test_dag = models.DAG(
        dag_id='test_dag',
        schedule_interval=self.schedule,
        start_date=utils.dates.days_ago(1))
    self.task = mock.Mock(models.BaseOperator, autospec=True)
    self.ml_engine_model_operator = mock.patch.object(
        mlengine_operator, 'MLEngineModelOperator', autospec=True).start()
    self.ml_engine_version_operator = mock.patch.object(
        mlengine_operator, 'MLEngineVersionOperator', autospec=True).start()
    self.branch_python_operator = mock.patch.object(
        python_operator, 'BranchPythonOperator', autospec=True).start()
    self.ml_engine_model_operator.return_value = self.task
    self.ml_engine_version_operator.return_value = self.task
    self.branch_python_operator.return_value = self.task

  def test_create_get_model_task(self):
    task = model_deployer_pipeline._create_get_model_task(
        self.project_id, self.model_name, self.test_dag)

    self.ml_engine_model_operator.assert_called_once_with(
        task_id='get_model',
        model={'name': self.model_name},
        project_id=self.project_id,
        operation='get',
        dag=self.test_dag)
    self.assertEqual(task, self.task)

  def test_create_model_task(self):
    task = model_deployer_pipeline._create_model_creator_task(
        self.project_id, self.model_name, self.test_dag)

    self.ml_engine_model_operator.assert_called_once_with(
        task_id='create_model',
        model={'name': self.model_name},
        project_id=self.project_id,
        operation='create',
        dag=self.test_dag)
    self.assertEqual(task, self.task)

  @mock.patch.object(datetime, 'datetime', autospec=True)
  def test_create_model_version_task(self, mock_datetime):
    time = '2019_01_01_01_01_01'
    mock_datetime.now.return_value.strftime.return_value = time

    task = model_deployer_pipeline._create_model_version_task(
        self.project_id, self.model_name, self.deployment_uri, self.test_dag)

    self.ml_engine_version_operator.assert_called_once_with(
        task_id='deploy_model',
        model_name=self.model_name,
        project_id=self.project_id,
        operation='create',
        version={
            'name': f'v_{time}',
            'deploymentUri': self.deployment_uri,
            'runtimeVersion': '1.14',
            'pythonVersion': '3.5',
            'framework': 'TENSORFLOW',
        },
        trigger_rule=trigger_rule.TriggerRule.ONE_SUCCESS,
        dag=self.test_dag)
    self.assertEqual(task, self.task)

  def test_create_branch_task(self):
    task = model_deployer_pipeline._create_branch_task(self.test_dag)

    self.branch_python_operator.assert_called_once_with(
        task_id='branching',
        python_callable=model_deployer_pipeline._decide_next_task,
        provide_context=True,
        dag=self.test_dag)
    self.assertEqual(task, self.task)

  def test_decide_next_task_creates_model(self):
    mock_task_instance = mock.Mock(models.BaseOperator, autospec=True)
    mock_task_instance.xcom_pull.return_value = None

    task_id = model_deployer_pipeline._decide_next_task(ti=mock_task_instance)

    self.assertEqual(task_id, model_deployer_pipeline._CREATE_MODEL_TASK_ID)

  def test_decide_next_task_skips_model_creation(self):
    mock_task_instance = mock.Mock(models.BaseOperator, autospec=True)
    mock_task_instance.xcom_pull.return_value = {'name': self.model_name}

    task_id = model_deployer_pipeline._decide_next_task(ti=mock_task_instance)

    self.assertEqual(task_id,
                     model_deployer_pipeline._SKIP_CREATE_MODEL_TASK_ID)

  @mock.patch.object(dummy_operator, 'DummyOperator', autospec=True)
  def test_create_dag(self, mock_dummy_operator):
    """Tests that returned DAG contains correct tasks."""
    get_model_task = mock.Mock(models.BaseOperator, autospec=True)
    create_model_task = mock.Mock(models.BaseOperator, autospec=True)
    branch_task = mock.Mock(models.BaseOperator, autospec=True)
    create_version_task = mock.Mock(models.BaseOperator, autospec=True)
    skip_create_model_task = mock.Mock(models.BaseOperator, autospec=True)
    self.ml_engine_model_operator.side_effect = [
        get_model_task, create_model_task
    ]
    self.ml_engine_version_operator.return_value = create_version_task
    self.branch_python_operator.return_value = branch_task
    mock_dummy_operator.return_value = skip_create_model_task

    actual_dag = model_deployer_pipeline.create_dag()

    self.assertIsInstance(actual_dag, models.DAG)
    get_model_task.set_downstream.assert_called_once_with(branch_task)
    branch_task.set_downstream.assert_called_once_with(
        [create_model_task, skip_create_model_task])
    skip_create_model_task.set_downstream.assert_called_once_with(
        create_version_task)
    create_model_task.set_downstream.assert_called_once_with(
        create_version_task)


if __name__ == '__main__':
  unittest.main()
