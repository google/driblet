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
"""Airflow subdag module to split BigQuery dataset into train/eval/test."""

import os
from typing import Dict, Union

from airflow import configuration
from airflow import models
from airflow.contrib.operators import bigquery_operator

from gps_building_blocks.ml import utils
from . import constants
from . import utils as dag_utils

SUB_DAG_ID = 'dataset-split'
WRITE_DISPOSITION = 'WRITE_TRUNCATE'
USE_LEGACY_SQL = False


def create_dataset_split_proportions(
    config: Dict[str, str]) -> Dict[str, Dict[str, Union[int, str]]]:
  """Creates dataset split proportions for train/eval and test datasets.

  Args:
    config: Configuration containing dataset destination tables and proportions.

  Returns:
    Mapped configurations for each of the three datasets.
  """
  train_proportion = int(config.get('train_proportion')) or 80
  eval_proportion = int(config.get('eval_proportion')) or 10
  test_proportion = int(config.get('test_proportion')) or 10
  total_proportion = sum([train_proportion, eval_proportion, test_proportion])

  # Check if sum of proportions are no bigger than 100.
  assert total_proportion == 100, (
      'Proportion sum is {total_proportion}, but needs to be exactly 100.')

  split_proportions = {
      'train': {
          'dest_table': config['train_dest_table'],
          'lower_bound': 0,
          'upper_bound': train_proportion
      },
      'eval': {
          'dest_table': config['eval_dest_table'],
          'lower_bound': train_proportion,
          'upper_bound': sum([train_proportion, eval_proportion])
      },
      'test': {
          'dest_table': config['test_dest_table'],
          'lower_bound': sum([train_proportion, eval_proportion]),
          'upper_bound': total_proportion
      }
  }
  return split_proportions


def create_dag(parent_dag_id: str) -> models.DAG:
  """Creates DAG for splitting data into train/eval/test datasets.

  Args:
    parent_dag_id: Id of the parent DAG.

  Returns:
      airflow.models.DAG: The DAG object.
  """
  base_config = dag_utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  dataset_split_config = dag_utils.get_airflow_variable_as_dict(
      constants.DATASET_SPLIT_CONFIG)

  # By convention, a SubDAG's name should be prefixed by its parent and a dot.
  dag_id = f'{parent_dag_id}.{SUB_DAG_ID}'
  dag_schedule_interval = base_config['schedule_interval']
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY
  dag = dag_utils.create_dag(dag_id, dag_schedule_interval, dag_retries,
                             dag_retry_delay)
  dag_dir = configuration.get('core', 'dags_folder')

  sql_path = os.path.join(dag_dir, 'queries/dataset_split.sql')
  id_column = dataset_split_config['id_column']
  input_table = dataset_split_config['input_table']
  input_table = f'{input_table}_{constants.TRAINING_SUFFIX}'
  query_params = {'bq_input_table': input_table, 'id_column': id_column}

  split_proportions = create_dataset_split_proportions(dataset_split_config)
  for dataset, config in split_proportions.items():
    query_params.update({
        'proportion_lower_bound': config['lower_bound'],
        'proportion_upper_bound': config['upper_bound']
    })
    sql = utils.configure_sql(sql_path, query_params)

    _ = bigquery_operator.BigQueryOperator(
        task_id=f'{SUB_DAG_ID}-{dataset}',
        sql=sql,
        destination_dataset_table=config['dest_table'],
        write_disposition=WRITE_DISPOSITION,
        use_legacy_sql=USE_LEGACY_SQL,
        dag=dag)
  return dag
