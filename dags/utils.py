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
"""Test utility functions for dags and subdags."""

import datetime
import json
from typing import Any, Dict, Optional, Union

from airflow import models
from airflow import utils
from airflow.operators import dummy_operator


def create_dag(dag_id: str,
               schedule: Union[str, None],
               retries: int,
               retry_delay: int,
               start_days_ago: int = 1,
               local_macros: Optional[Dict[str, Any]] = None,
               **kwargs) -> models.DAG:
  """Creates Airflow DAG with appropriate default args.

  Args:
    dag_id: Id for the DAG.
    schedule: DAG run schedule. Ex: if set `@once`, DAG will be scheduled to run
      only once. For more refer:
        https://airflow.apache.org/docs/stable/scheduler.html
    retries: How many times DAG retries.
    retry_delay: The interval (in minutes) to trigger the retry.
    start_days_ago: Start date of the DAG. By default it's set to 1 (yesterday)
      to trigger DAG as soon as its deployed.
    local_macros: A dictionary of macros that will be exposed in jinja
      templates.
    **kwargs: Keyword arguments.

  Returns:
    Instance of airflow.models.DAG.
  """
  default_args = {
      'retries': retries,
      'retry_delay': datetime.timedelta(minutes=retry_delay),
      'start_date': utils.dates.days_ago(start_days_ago)
  }

  if kwargs:
    default_args.update(kwargs)

  return models.DAG(
      dag_id=dag_id,
      schedule_interval=schedule,
      user_defined_macros=local_macros,
      default_args=default_args)


def get_airflow_variable_as_dict(
    key: str) -> Dict[str, Union[str, Dict[str, str]]]:
  """Retrieves Airflow variables given key.

  Args:
    key: Airflow variable key.

  Returns:
    Airflow Variable value as a string or Dict.

  Raises:
    Exception if given Airflow Variable value cannot be parsed.
  """
  value = models.Variable.get(key)
  try:
    value_dict = json.loads(value)
  except json.decoder.JSONDecodeError as error:
    raise Exception('Provided key "{}" cannot be decoded. {}'.format(
        key, error))
  return value_dict


def create_test_dag(
    dag_id: str = 'test_dag',
    dag: Optional[models.DAG] = None,
    task_id: str = 'test_task',
) -> Union[models.DAG, dummy_operator.DummyOperator]:
  """Creates Airflow sample DAG for testing.

  Args:
    dag_id: Dag id.
    dag: An instance of DAG.
    task_id: Operator task id.

  Returns:
    dag: An instance of DAG.
  """
  if not dag:
    dag = models.DAG(
        dag_id=f'{dag_id}',
        schedule_interval='0 1 * * *',
        start_date=datetime.datetime(2021, 10, 2))
  _ = dummy_operator.DummyOperator(task_id=task_id, dag=dag)
  return dag


def create_test_operator(
    dag_id: str = 'test_dag',
    dag: Optional[models.DAG] = None,
    task_id: str = 'test_task',
) -> Union[models.DAG, dummy_operator.DummyOperator]:
  """Creates Airflow sample Dummy operator for testing.

  Args:
    dag_id: Dag id.
    dag: An instance of DAG.
    task_id: Operator task id.

  Returns:
    An instance of DummyOperator.
  """
  if not dag:
    dag = models.DAG(
        dag_id=f'{dag_id}',
        schedule_interval='0 1 * * *',
        start_date=datetime.datetime(2021, 10, 2))
  return dummy_operator.DummyOperator(task_id=task_id, dag=dag)
