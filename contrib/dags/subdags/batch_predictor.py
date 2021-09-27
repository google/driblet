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
"""Airflow subdag module to do batch prediction."""

import datetime
import os

from airflow import models
from airflow.contrib.operators import mlengine_operator
import pytz

from driblet.dags import constants
from driblet.dags import utils

# Airflow configuration.
SUB_DAG_ID = 'batch-predictor'

# Configs for MLEngineBatchPredictionOperator.
_INPUT_DATA_FORMAT = 'TF_RECORD'
_GCP_CONN_ID = 'google_cloud_default'


def create_dag(parent_dag_id: str) -> models.DAG:
  """Creates DAG for batch prediction.

  Args:
    parent_dag_id: Id of the parent DAG.

  Returns:
      airflow.models.DAG: The DAG object.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  gcp_config = utils.get_airflow_variable_as_dict(constants.GCP_CONFIG)
  predict_config = utils.get_airflow_variable_as_dict(
      constants.PREDICTION_CONFIG)
  transform_config = utils.get_airflow_variable_as_dict(
      constants.TRANSFORM_CONFIG)

  project_id = gcp_config['project_id']
  region = gcp_config['region']
  storage_bucket = gcp_config['storage_bucket']

  # By convention, a SubDAG's name should be prefixed by its parent and a dot.
  dag_id = f'{parent_dag_id}.{SUB_DAG_ID}'
  dag_schedule_interval = base_config['schedule_interval']
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY

  dag = utils.create_dag(dag_id, dag_schedule_interval, dag_retries,
                         dag_retry_delay)

  local_timezone = pytz.timezone(base_config['local_timezone'])
  today = datetime.datetime.now(local_timezone)
  date_suffix = today.strftime(constants.DATE_FORMAT)
  job_id = f'driblet_predict_{date_suffix}'

  predict_file_prefix = f'{constants.PREDICT_TFRECORD_PREFIX}-*'
  transform_output_dir = transform_config['output_dir']
  input_path = os.path.join(storage_bucket, transform_output_dir,
                            constants.PREDICTION_SUFFIX, predict_file_prefix)
  output_path = os.path.join(storage_bucket, predict_config['output_dir'],
                             date_suffix)

  model_name = predict_config['model_name']
  model_version = predict_config['model_version']

  _ = mlengine_operator.MLEngineBatchPredictionOperator(
      task_id=SUB_DAG_ID,
      project_id=project_id,
      job_id=job_id,
      data_format=_INPUT_DATA_FORMAT,
      input_paths=[input_path],
      output_path=output_path,
      region=region,
      model_name=model_name,
      version_name=model_version,
      gcp_conn_id=_GCP_CONN_ID,
      dag=dag)

  return dag
