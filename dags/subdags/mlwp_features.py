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
"""ML Windowing Pipeline based feature generator."""

from typing import Optional

from airflow import models
from airflow.operators import python_operator

from gps_building_blocks.ml.data_prep.ml_windowing_pipeline import ml_windowing_pipeline
from . import constants
from . import utils

# Airflow configuration.
SUB_DAG_ID = 'mlwp_features'
PREDICTION_TASK_ID = 'mlwp_prediction_features'
TRAIN_TASK_ID = 'mlwp_training_features'


def create_operator(
    parent_dag_id: str,
    pipeline_mode: constants.PipelineMode,
    dag: Optional[models.DAG] = None,
) -> python_operator.PythonOperator:
  """Creates ML Windowing Pipeline task to generate training dataset.

  This task requires multiple parameters. Refer to
  https://github.com/google/gps_building_blocks/tree/master/py/gps_building_blocks/ml/data_prep/ml_windowing_pipeline#step-3-run-data-windowing-pipeline
  for details.

  Args:
    parent_dag_id: Id of the parent DAG.
    pipeline_mode: Pipeline running mode. Should be either TRAIN or PREDICT.
    dag: An instance of models.DAG.

  Returns:
      airflow.models.DAG: The DAG object.

  Raises:
    ValueError if PipelineMode is incorrect.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  gcp_config = utils.get_airflow_variable_as_dict(constants.GCP_CONFIG)

  project_id = gcp_config['project_id']
  # By convention, a SubDAG's name should be prefixed by its parent and a dot.

  if not dag:
    dag_id = f'{parent_dag_id}.{SUB_DAG_ID}'
    dag_schedule_interval = base_config['schedule_interval']
    dag_retries = constants.DAG_RETRIES
    dag_retry_delay = constants.DAG_RETRY_DELAY
    dag = utils.create_dag(dag_id, dag_schedule_interval, dag_retries,
                           dag_retry_delay)

  mwp_config = utils.get_airflow_variable_as_dict(constants.MWP_CONFIG)
  params = {
      'project_id':
          project_id,
      'dataset_id':
          mwp_config['dataset_id'],
      'run_id':
          mwp_config['run_id'],
      'analytics_table':
          mwp_config['analytics_table'],
      'slide_interval_in_days':
          mwp_config['slide_interval_in_days'],
      'lookback_window_size_in_days':
          mwp_config['lookback_window_size_in_days'],
      'lookback_window_gap_in_days':
          mwp_config['lookback_window_gap_in_days'],
      'prediction_window_gap_in_days':
          mwp_config['prediction_window_gap_in_days'],
      'prediction_window_size_in_days':
          mwp_config['prediction_window_size_in_days'],
      'sum_values':
          mwp_config['sum_values'],
      'avg_values':
          mwp_config['avg_values'],
      'count_values':
          mwp_config['count_values'],
      'mode_values':
          mwp_config['mode_values'],
      'proportions_values':
          mwp_config['proportions_values'],
      'latest_values':
          mwp_config['latest_values']
  }

  if pipeline_mode == constants.PipelineMode.PREDICT:
    python_callable = ml_windowing_pipeline.run_prediction_pipeline
    task_id = PREDICTION_TASK_ID
  elif pipeline_mode == constants.PipelineMode.TRAIN:
    python_callable = ml_windowing_pipeline.run_features_pipeline
    task_id = TRAIN_TASK_ID
  else:
    raise ValueError('Pipeline mode is incorrect. Provide either "train" or '
                     '"predict".')

  return python_operator.PythonOperator(
      task_id=task_id,
      python_callable=python_callable,
      op_args={'params': params},
      dag=dag,
  )
