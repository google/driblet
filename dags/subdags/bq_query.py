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
"""Airflow subdag module to execute a custom query in BigQuery."""

from typing import Dict, Optional

from airflow import models
from airflow.contrib.operators import bigquery_operator

from gps_building_blocks.ml import utils
from . import constants
from . import utils as dag_utils

# Airflow configuration.
SUB_DAG_ID = 'bq-query'
TASK_ID = 'custom-query-task'


def create_operator(
    parent_dag_id: str,
    sql_params: Dict[str, str],
    sql_path: str,
    task_id: str = TASK_ID,
    dag: Optional[models.DAG] = None,
    bigquery_conn_id: str = 'bigquery-default'
) -> bigquery_operator.BigQueryOperator:
  """Creates DAG to execute a custom query in BigQuery.

  Args:
    parent_dag_id: Id of the parent DAG.
    sql_params: SQL query parameters to used to replace in SQL scripts.
    sql_path: File path of SQL to execute.
    task_id: BigQueryOperator task id.
    dag: An instance of models.DAG.
    bigquery_conn_id: The connection ID to connect to Google Cloud Platform.

  Returns:
    Instance of bigquery_operator.BigQueryOperator.

  Raises:
    KeyError if there is missing key in input configuration.
  """
  if not dag:
    base_config = dag_utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
    dag_id = f'{parent_dag_id}.{SUB_DAG_ID}'
    dag_schedule_interval = base_config['schedule_interval']
    dag_retries = constants.DAG_RETRIES
    dag_retry_delay = constants.DAG_RETRY_DELAY
    dag = dag_utils.create_dag(dag_id, dag_schedule_interval, dag_retries,
                               dag_retry_delay)

  # Parse SQL.
  try:
    sql = utils.configure_sql(sql_path, sql_params)
  except KeyError as error:
    raise KeyError('Missing key in input configuration: ' + str(error))

  return bigquery_operator.BigQueryOperator(
      task_id=task_id,
      sql=sql,
      use_legacy_sql=False,
      bigquery_conn_id=bigquery_conn_id,
      dag=dag)
