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
"""Airflow subdag module to delete files & folders from Cloud Storage bucket."""

import os
from typing import List
from airflow import models
from google.cloud import storage
from driblet.contrib.plugins.operators import gcs_delete_blob_operator
from driblet.dags import constants
from driblet.dags import utils

# Airflow configuration.
SUB_DAG_ID = 'storage-cleaner'


def create_dag(parent_dag_id: str, prefixes: List[str]) -> models.DAG:
  """Creates DAG to remove files and folders from Cloud Storage bucket.

  Args:
    parent_dag_id: Id of the parent DAG.
    prefixes: List of Cloud storage folders to remove files from.

  Returns:
      airflow.models.DAG: The DAG object.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  gcp_config = utils.get_airflow_variable_as_dict(constants.GCP_CONFIG)

  project_id = gcp_config['project_id']
  storage_bucket = gcp_config['storage_bucket']
  # By convention, a SubDAG's name should be prefixed by its parent and a dot.
  dag_id = f'{parent_dag_id}.{SUB_DAG_ID}'
  dag_schedule_interval = base_config['schedule_interval']
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY

  dag = utils.create_dag(dag_id, dag_schedule_interval, dag_retries,
                         dag_retry_delay)

  client = storage.Client(project=project_id)
  gcs_bucket_name = os.path.basename(storage_bucket)

  _ = gcs_delete_blob_operator.GCSDeleteBlobOperator(
      task_id=SUB_DAG_ID,
      client=client,
      gcs_bucket=gcs_bucket_name,
      prefixes=prefixes,
      dag=dag)

  return dag
