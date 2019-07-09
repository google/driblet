# coding=utf-8
# Copyright 2019 Google LLC
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
"""Airflow DAG for Driblet workflow.

This DAG relies on multiple Airflow variables
https://airflow.apache.org/concepts.html#variables

* project_id - Cloud Project ID to use for BigQuery and Cloud Storage.
* bucket_name - Cloud Storage bucket to write/read data.
* bq_dataset - BigQuery dataset name that holds table used for prediction.
* bq_input_table - BigQuery table storing data used by ML Engine for prediction.
* bq_output_table - BigQuery table that will store predicted output.
* model_name - ML Engine model name used for prediction.
* model_version - ML Engine deployed model version.
"""

import datetime
import os

from airflow import configuration
from airflow import models
from airflow import utils
from airflow.contrib.operators import dataflow_operator
from airflow.contrib.operators import mlengine_operator
from airflow.operators.gcp_plugins import GCSDeleteBlobOperator
from airflow.operators.gcp_plugins import GCStoBQOperator

from google.cloud import bigquery
from google.cloud import storage

_AIRFLOW_ENV = 'AIRFLOW_HOME'
_DAG_NAME = 'driblet_dag'
_DAG_SCHEDULE = 1  # In days
_DAG_RETRIES = 0  # DAG retries
_DAG_RETRY_DELAY = 5  # In minutes


def initialize_dag():
  """Returns an Airflow directed acyclic graph with appropriate default args."""
  default_dag_args = {
      # Setting start date to yesterday triggers DAG immediately after being
      # deployed.
      'start_date': utils.dates.days_ago(_DAG_SCHEDULE),  # Yesterday
      'retries': _DAG_RETRIES,
      'retry_delay': datetime.timedelta(minutes=_DAG_RETRY_DELAY)
  }

  return models.DAG(
      dag_id=_DAG_NAME,
      schedule_interval=datetime.timedelta(days=_DAG_SCHEDULE),
      default_args=default_dag_args)


def get_env_variables():
  """Obtains Airflow models variables that are set on environment variables.

  Returns:
    env_variables: Dictionary of Airflow environment variables.
  """
  variable_keys = [
      'project_id', 'location', 'region', 'bucket_name', 'bq_dataset',
      'bq_input_table', 'bq_output_table', 'dataset_expiration', 'model_name',
      'model_version'
  ]
  env_variables = {}
  for key in variable_keys:
    env_variables[key] = models.Variable.get(key)
  return env_variables


def create_dag(env_variables):
  """Creates the Airflow directed acyclic graph.

  Args:
    env_variables: Dictionary of Airflow environment variables.

  Returns:
    driblet_dag: An instance of models.DAG.
  """
  driblet_dag = initialize_dag()

  # Clients setup.
  project_id = env_variables['project_id']
  bq_client = bigquery.Client(project=project_id)
  gcs_client = storage.Client(project=project_id)

  # TASK 1: Convert BigQuery CSV to TFRECORD.
  dag_dir = configuration.get('core', 'dags_folder')
  transformer_py = os.path.join(dag_dir, 'tasks/preprocess', 'transformer.py')
  bq_to_tfrecord = dataflow_operator.DataFlowPythonOperator(
      task_id='bq-to-tfrecord',
      py_file=transformer_py,
      options={
          'project':
              project_id,
          'predict-data':
              '{}.{}.{}_{}'.format(project_id, env_variables['bq_dataset'],
                                   env_variables['bq_input_table'],
                                   datetime.datetime.now().strftime('%Y%m%d')),
          'data-source':
              'bigquery',
          'transform-dir':
              'gs://%s/transformer' % env_variables['bucket_name'],
          'output-dir':
              'gs://%s/input' % env_variables['bucket_name'],
          'mode':
              'predict'
      },
      dataflow_default_options={'project': project_id},
      dag=driblet_dag)

  # TASK 2: Make prediction from CSV in GCS.
  make_predictions = mlengine_operator.MLEngineBatchPredictionOperator(
      task_id='make-predictions',
      project_id=project_id,
      job_id='driblet_run_{}'.format(
          datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')),
      data_format='TF_RECORD',
      input_paths=['gs://%s/input/predict-*' % env_variables['bucket_name']],
      output_path='gs://%s/output' % env_variables['bucket_name'],
      region=env_variables['region'],
      model_name=env_variables['model_name'],
      version_name=env_variables['model_version'],
      gcp_conn_id='google_cloud_default',
      dag=driblet_dag)

  # TASK 3: Export predicted CSV from Cloud Storage to BigQuery.
  job_config = bigquery.LoadJobConfig()
  job_config.autodetect = True
  job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
  job_config.time_partitioning = bigquery.TimePartitioning(
      type_=bigquery.TimePartitioningType.DAY,  # Sets daily partitioned table.
      expiration_ms=env_variables['dataset_expiration'])
  gcs_to_bigquery = GCStoBQOperator(
      task_id='gcs-to-bigquery',
      bq_client=bq_client,
      gcs_client=gcs_client,
      job_config=job_config,
      dataset_id=env_variables['bq_dataset'],
      table_id=env_variables['bq_output_table'],
      gcs_bucket=env_variables['bucket_name'],
      gcs_location=env_variables['location'],
      exclude_prefix='errors_stats',  # Exclude files starting with name.
      dir_prefix='output',
      dag=driblet_dag)

  # TASK 4: Delete files in Cloud Storage bucket.
  gcs_delete_blob = GCSDeleteBlobOperator(
      task_id='gcs-delete-blob',
      client=gcs_client,
      gcs_bucket=env_variables['bucket_name'],
      prefixes=['input', 'output'],
      dag=driblet_dag)

  make_predictions.set_upstream(bq_to_tfrecord)
  make_predictions.set_downstream(gcs_to_bigquery)
  gcs_delete_blob.set_upstream(gcs_to_bigquery)

  return driblet_dag


# By design Airflow expects DAG as a global variable
# (refer to https://airflow.apache.org/concepts.html#scope).
# At the same time, we need to be able to import this file as a module
# (e.g. for testing), in which case we dont want immediate execution.
# This logic is contolled by environment variable.
if os.getenv(_AIRFLOW_ENV):
  variables = get_env_variables()
  dag = create_dag(variables)
