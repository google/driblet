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
"""Main Airflow DAG for Driblet v.2 training pipeline.

TODO(): Separate data generator pipeline from training and prediction
pipelines. This will help to use one data pipeline for both prediction and
training.
"""

import os
from typing import List

from airflow import models
from airflow.operators import subdag_operator

from driblet.contrib.dags.subdags import model_trainer
from driblet.contrib.dags.subdags import storage_cleaner
from driblet.dags import constants
from driblet.dags import utils

_DAG_ID = constants.TRAINING_DAG_ID
_DAG_SCHEDULE = '@once'


def create_storage_cleaner_task(
    main_dag: models.DAG,
    prefixes: List[str]) -> subdag_operator.SubDagOperator:
  """Creates Cloud Storage cleaner SubDag task.

  Args:
    main_dag: The models.DAG instance.
    prefixes: List of Cloud storage folders to remove files from.

  Returns:
    subdag_operator.SubDagOperator.
  """
  storage_cleaner_subdag = storage_cleaner.create_dag(
      parent_dag_id=_DAG_ID, prefixes=prefixes)

  return subdag_operator.SubDagOperator(
      task_id=storage_cleaner.SUB_DAG_ID,
      subdag=storage_cleaner_subdag,
      dag=main_dag)


def create_training_task(
    main_dag: models.DAG) -> subdag_operator.SubDagOperator:
  """Creates training SubDag.

  Args:
    main_dag: The models.DAG instance.

  Returns:
    subdag_operator.SubDagOperator: Training task.
  """

  training_subdag = model_trainer.create_dag(parent_dag_id=_DAG_ID)

  return subdag_operator.SubDagOperator(
      task_id=model_trainer.SUB_DAG_ID,
      subdag=training_subdag,
      dag=main_dag,
  )


def create_dag() -> models.DAG:
  """Creates Airflow DAG for prediction pipeline.

  Returns:
    main_dag: An instance of models.DAG.
  """
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY
  main_dag = utils.create_dag(_DAG_ID, _DAG_SCHEDULE, dag_retries,
                              dag_retry_delay)
  trainig_config = utils.get_airflow_variable_as_dict(constants.TRAINING_CONFIG)
  training_job_dir_path = trainig_config['job_dir']

  storage_cleaner_task = create_storage_cleaner_task(
      main_dag, prefixes=[training_job_dir_path])
  training_task = create_training_task(main_dag)
  storage_cleaner_task.set_downstream(training_task)

  return main_dag


if os.getenv(constants.AIRFLOW_ENV):
  dag = create_dag()
