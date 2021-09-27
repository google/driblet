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
"""Airflow DAG to deploy ML model on AI platform.

This DAG deploys trained ML model using Airflow operators so that the
predictions can be performed.
"""
import datetime
import os
from typing import Any

from airflow import models
from airflow.contrib.operators import mlengine_operator
from airflow.operators import dummy_operator
from airflow.operators import python_operator
from airflow.utils import trigger_rule

from driblet.dags import constants
from driblet.dags import utils

# DAG and Task IDs.
_DAG_ID = constants.MODEL_DEPLOY_DAG_ID
_GET_MODEL_TASK_ID = 'get_model'
_CREATE_MODEL_TASK_ID = 'create_model'
_DEPLOY_MODEL_TASK_ID = 'deploy_model'
_SKIP_CREATE_MODEL_TASK_ID = 'skip_create_model'
_RUNTIME_VERSION = '1.14'
_PYTHON_VERSION = '3.5'
_FRAMEWORK = 'TENSORFLOW'


def _create_mlengine_model_operator_task(
    project_id: str, model_name: str, task_id: str, operation: str,
    main_dag: models.DAG) -> mlengine_operator.MLEngineModelOperator:
  """Creates ML Engine model operator task.

  Args:
    project_id: Google Cloud Platform project id.
    model_name: Name of the ML model.
    task_id: Id of the task.
    operation: The operation to perform.
    main_dag: The models.DAG instance.

  Returns:
    Model operator task.
  """
  return mlengine_operator.MLEngineModelOperator(
      task_id=task_id,
      model={'name': model_name},
      project_id=project_id,
      operation=operation,
      dag=main_dag)


def _create_get_model_task(
    project_id: str, model_name: str,
    main_dag: models.DAG) -> mlengine_operator.MLEngineModelOperator:
  """Creates ML Engine get model task.

  Args:
    project_id: Google Cloud Platform project id.
    model_name: Name of the ML model.
    main_dag: The models.DAG instance.

  Returns:
    Get model task.
  """
  operation = 'get'
  return _create_mlengine_model_operator_task(project_id, model_name,
                                              _GET_MODEL_TASK_ID, operation,
                                              main_dag)


def _create_model_creator_task(
    project_id: str, model_name: str,
    main_dag: models.DAG) -> mlengine_operator.MLEngineModelOperator:
  """Creates ML Engine model creator task.

  Args:
    project_id: Google Cloud Platform project id.
    model_name: Name of the ML model.
    main_dag: The models.DAG instance.

  Returns:
    Create model task.
  """
  operation = 'create'
  return _create_mlengine_model_operator_task(project_id, model_name,
                                              _CREATE_MODEL_TASK_ID, operation,
                                              main_dag)


def _create_model_version_task(
    project_id: str, model_name: str, deployment_uri: str,
    main_dag: models.DAG) -> mlengine_operator.MLEngineModelOperator:
  """Creates ML Engine model version task.

  Args:
    project_id: Google Cloud Platform project id.
    model_name: Name of the ML model.
    deployment_uri: The Cloud Storage path of trained Tensorflow model.
    main_dag: The models.DAG instance.

  Returns:
    Model version task.
  """
  now = datetime.datetime.now()
  return mlengine_operator.MLEngineVersionOperator(
      task_id=_DEPLOY_MODEL_TASK_ID,
      project_id=project_id,
      model_name=model_name,
      version={
          'name': 'v_' + now.strftime('%Y-%m-%d_%H-%M-%S'),
          'deploymentUri': deployment_uri,
          'runtimeVersion': _RUNTIME_VERSION,
          'pythonVersion': _PYTHON_VERSION,
          'framework': _FRAMEWORK,
      },
      operation='create',
      trigger_rule=trigger_rule.TriggerRule.ONE_SUCCESS,
      dag=main_dag)


def _decide_next_task(**kwargs: Any) -> str:
  """Decides which task to execute next.

  Args:
    **kwargs: Dictionary containing task instance details.

  Returns:
    Task id of the next task.
  """
  task_instance = kwargs['ti']
  get_model_data = task_instance.xcom_pull(task_ids=_GET_MODEL_TASK_ID)
  return _SKIP_CREATE_MODEL_TASK_ID if get_model_data else _CREATE_MODEL_TASK_ID


def _create_branch_task(
    main_dag: models.DAG) -> python_operator.BranchPythonOperator:
  """Create branch task for creating ML model.

  If the ML model already exists, there is no reason to create the model.

  Args:
    main_dag: The models.DAG instance.

  Returns:
    Branch task.
  """
  return python_operator.BranchPythonOperator(
      task_id='branching',
      python_callable=_decide_next_task,
      provide_context=True,
      dag=main_dag)


def create_dag() -> models.DAG:
  """Creates DAG for deploying TensorFlow model to GCP AI Platform.

  Returns:
      The DAG object.
  """
  # Get airflow configurations.
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  gcp_config = utils.get_airflow_variable_as_dict(constants.GCP_CONFIG)
  model_deployer_config = utils.get_airflow_variable_as_dict(
      constants.MODEL_DEPLOY_CONFIG)

  schedule = base_config['schedule_interval']
  retries = constants.DAG_RETRIES
  retry_delay = constants.DAG_RETRY_DELAY
  project_id = gcp_config['project_id']
  model_name = model_deployer_config['model_name']
  deployment_uri = model_deployer_config['deployment_uri']
  # By convention, a SubDAG's name should be prefixed by its parent and a dot.
  main_dag = utils.create_dag(_DAG_ID, schedule, retries, retry_delay)

  # Create all the tasks.
  get_model_task = _create_get_model_task(project_id, model_name, main_dag)
  create_model_task = _create_model_creator_task(project_id, model_name,
                                                 main_dag)
  create_version_task = _create_model_version_task(project_id, model_name,
                                                   deployment_uri, main_dag)
  branch_task = _create_branch_task(main_dag)
  skip_create_model_task = dummy_operator.DummyOperator(
      task_id=_SKIP_CREATE_MODEL_TASK_ID, dag=main_dag)

  # Set task dependencies.
  get_model_task.set_downstream(branch_task)
  branch_task.set_downstream([create_model_task, skip_create_model_task])
  skip_create_model_task.set_downstream(create_version_task)
  create_model_task.set_downstream(create_version_task)

  return main_dag


if os.getenv(constants.AIRFLOW_ENV):
  dag = create_dag()
