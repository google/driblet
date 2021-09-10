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
"""Feature pipeline."""
import os
from typing import Union

from airflow import configuration
from airflow import models
from airflow.contrib.operators import bigquery_operator
from airflow.operators import python_operator

from gps_building_blocks.ml.data_prep.ml_windowing_pipeline import ml_windowing_pipeline
from . import constants
from . import utils
from .subdags import bq_query
from .subdags import mlwp_features

# Airflow configuration.
_DAG_ID = 'feature_pipeline'


def create_mlwp_feature_task(main_dag: models.DAG,
                             project_id: str) -> python_operator.PythonOperator:
  """Creates ML Data Windowing Pipeline task.

  This task requires multiple parameters. Refer to
  https://github.com/google/gps_building_blocks/tree/master/py/gps_building_blocks/ml/data_prep/ml_windowing_pipeline#step-3-run-data-windowing-pipeline
  for details.

  Args:
    main_dag: The models.DAG instance.
    project_id: GCP project id.

  Returns:
    python_operator.PythonOperator: ML windowing pipeline task.
  """
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
      'snapshot_start_date':
          mwp_config['snapshot_start_date'],
      'snapshot_end_date':
          mwp_config['snapshot_end_date'],
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
      'stop_on_first_positive':
          mwp_config['stop_on_first_positive'],
      'features_sql':
          mwp_config['features_sql'],
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

  return python_operator.PythonOperator(
      task_id='mlwp_feature_task',
      python_callable=ml_windowing_pipeline.run_end_to_end_pipeline,
      op_args={'params': params},
      dag=main_dag,
  )


def create_custom_feature_task(
    main_dag: models.DAG
) -> Union[models.DAG, bigquery_operator.BigQueryOperator]:
  """Creates a task to generate features based on custom SQL.

  Args:
    main_dag: The models.DAG instance.

  Returns:
    bigquery_operator.BigQueryOperator: Custom BigQuery dataset creation task.
  """
  custom_query_config = utils.get_airflow_variable_as_dict(
      constants.CUSTOM_QUERY_CONFIG)
  dag_dir = configuration.get('core', 'dags_folder')
  sql_path = os.path.join(dag_dir, constants.FEATURES_SQL_PATH)

  return bq_query.create_operator(
      task_id='custom_feature_task',
      parent_dag_id=_DAG_ID,
      sql_params=custom_query_config['params'],
      sql_path=sql_path,
      dag=main_dag)


def create_dag() -> models.DAG:
  """Creates Airflow DAG for prediction pipeline.

  Returns:
    main_dag: An instance of models.DAG.

  Raises:
    ValueError if unsupported feature generator is provided.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  dag_schedule_interval = base_config['schedule_interval']
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY
  main_dag = utils.create_dag(_DAG_ID, dag_schedule_interval, dag_retries,
                              dag_retry_delay)
  feature_generator = base_config['feature_generator']

  if feature_generator == constants.FeatureGenerator.MLWP.value:
    pipeline_mode_ = base_config['pipeline_mode']
    if str(pipeline_mode_).lower() in ['predict', 'prediction']:
      pipeline_mode = constants.PipelineMode.PREDICT
    else:
      pipeline_mode = constants.PipelineMode.TRAIN
    _ = mlwp_features.create_operator(
        parent_dag_id=_DAG_ID, dag=main_dag, pipeline_mode=pipeline_mode)
  elif feature_generator == constants.FeatureGenerator.CUSTOM.value:
    custom_query_config = utils.get_airflow_variable_as_dict(
        constants.CUSTOM_QUERY_CONFIG)
    dag_dir = configuration.get('core', 'dags_folder')
    sql_path = os.path.join(dag_dir, constants.FEATURES_SQL_PATH)

    _ = bq_query.create_operator(
        parent_dag_id=_DAG_ID,
        dag=main_dag,
        sql_params=custom_query_config['params'],
        sql_path=sql_path)
  else:
    raise ValueError(f'{feature_generator} is not supported.'
                     'Provide either "custom" or "mlwp".')

  return main_dag


if os.getenv(constants.AIRFLOW_ENV):
  dag = create_dag()
