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
"""Prediction pipeline.

This pipeline executes following two tasks:
1. Generate scoring dataset.
2. Batch score using pretrained BQML model.
"""

import os
from typing import Any, Dict, Optional, Union

from airflow import configuration
from airflow import models
from airflow.contrib.operators import bigquery_operator
from airflow.operators import python_operator
# from airflow.utils import helpers

from . import constants
from . import utils
from .subdags import bq_query
from .subdags import mlwp_features

# DAG configuration.
_DAG_ID = constants.PREDICTION_DAG_ID
# This DAG should not be scheduled as it will be tiggerred by
# `dataset_pipeline_dag`.
_DAG_SCHEDULE = None
# Dag directory containing subdags and custom SQL scripts.
_DAG_DIR = configuration.get('core', 'dags_folder')


def _create_prediction_feature_generator_task(
    main_dag: models.DAG,
    feature_generator: constants.FeatureGenerator,
    sql_params: Optional[Dict[str, Any]] = None
) -> Union[python_operator.PythonOperator, bigquery_operator.BigQueryOperator]:
  """Creates prediction feature generator task.

  Args:
    main_dag: The models.DAG instance.
    feature_generator: Either custom query or MLWP pipeline is used to generate
      prediction features.
    sql_params: Custom parameters to apply to features.sql script. It is
      required only in FeatureGenerator.CUSTOM mode.

  Returns:
    task: Either PyhonOperator or BigQueryOperator.
  """
  if feature_generator == constants.FeatureGenerator.MLWP:
    task = mlwp_features.create_operator(
        parent_dag_id=_DAG_ID,
        dag=main_dag,
        pipeline_mode=constants.PipelineMode.PREDICT)
  elif feature_generator == constants.FeatureGenerator.CUSTOM:
    assert sql_params is not None, ('Provide `sql_params` to apply to '
                                    'features.sql.')
    sql_path = os.path.join(_DAG_DIR, constants.FEATURES_SQL_PATH)

    task = bq_query.create_operator(
        parent_dag_id=_DAG_ID,
        dag=main_dag,
        sql_params=sql_params,
        sql_path=sql_path)
  else:
    raise ValueError(f'{feature_generator.value} is not supported.'
                     'Provide either "custom" or "mlwp".')
  return task


def _create_batch_predictor_task(
    main_dag: models.DAG,
    sql_params: Dict[str, str]) -> bigquery_operator.BigQueryOperator:
  """Creates batch predictor task based on BQML model.

  Args:
    main_dag: The models.DAG instance.
    sql_params: Custom parameters to apply to constants.PREDICTION_SQL_PATH
      script.

  Returns:
    Instance of bigquery_operator.BigQueryOperator.
  """
  sql_path = os.path.join(_DAG_DIR, constants.PREDICTION_SQL_PATH)

  return bq_query.create_operator(
      parent_dag_id=_DAG_ID,
      dag=main_dag,
      sql_params=sql_params,
      sql_path=sql_path)


def create_dag() -> models.DAG:
  """Creates Airflow DAG for prediction pipeline.

  This execture two workflows:
  1. Generate BigQuery table with scoring dataset.
  2. Execute BQML prediction job and save outputs as BigQuery table.

  Returns:
    main_dag: An instance of models.DAG.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  dag_schedule_interval = base_config['schedule_interval']
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY
  main_dag = utils.create_dag(_DAG_ID, dag_schedule_interval, dag_retries,
                              dag_retry_delay)
  prediction_config = utils.get_airflow_variable_as_dict(
      constants.PREDICTION_PIPELINE_CONFIG)
  mlwp = constants.FeatureGenerator.MLWP
  if str(prediction_config['feature_generator']).lower() == mlwp.value:
    feature_generator = mlwp
  else:
    feature_generator = constants.FeatureGenerator.CUSTOM

  feature_generator_task = _create_prediction_feature_generator_task(
      main_dag=main_dag,
      feature_generator=feature_generator,
      sql_params=prediction_config['features_sql_params'])
  batch_predictor_task = _create_batch_predictor_task(
      main_dag=main_dag, sql_params=prediction_config['prediction_sql_params'])
  feature_generator_task.set_downstream(batch_predictor_task)

  return main_dag


if os.getenv(constants.AIRFLOW_ENV):
  dag = create_dag()
