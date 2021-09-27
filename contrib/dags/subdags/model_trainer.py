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
"""Airflow DAG to run training on gcloud AI platform.

The MLEngineTrainingOperator submits the training job to AI platform with given
arguments.
"""
import os
import uuid
from airflow import configuration
from airflow import models
from airflow.contrib.operators import mlengine_operator

from driblet.dags import constants
from driblet.dags import utils

# Name of the SubDAG.
SUB_DAG_ID = 'training'

# Package configs.
_GCP_CONN_ID = 'google_cloud_default'
_MODE = 'CLOUD'
_PYTHON_VERSION = '3.5'
_RUNTIME_VERSION = '1.14'
_TRAINING_MODULE = 'model.trainer'
_TRAINING_PACKAGE = 'model'


def create_dag(parent_dag_id: str,
               gcp_conn_id: str = _GCP_CONN_ID,
               mode: str = _MODE) -> models.DAG:
  """Creates DAG to run feature transform on DataFlow.

  Args:
    parent_dag_id: Name of the parent DAG.
    gcp_conn_id: GCP connection ID to access AI platform.
    mode: Can be one of ‘DRY_RUN’/’CLOUD’. In ‘DRY_RUN’ mode, no real training
      job will be launched, but the MLEngine training job request will be
      printed out. In ‘CLOUD’ mode, a real MLEngine training job creation
      request will be issued.

  Returns:
      airflow.models.DAG: The DAG object.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  gcp_config = utils.get_airflow_variable_as_dict(constants.GCP_CONFIG)
  transform_config = utils.get_airflow_variable_as_dict(
      constants.TRANSFORM_CONFIG)
  training_config = utils.get_airflow_variable_as_dict(
      constants.TRAINING_CONFIG)

  train_file_suffix = constants.TRAIN_TFRECORD_SUFFIX
  eval_file_suffix = constants.EVAL_TFRECORD_SUFFIX
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
  dag_dir = configuration.get('core', 'dags_folder')

  transform_dir = os.path.join(storage_bucket,
                               transform_config['transform_dir'])
  transform_output_dir = os.path.join(storage_bucket,
                                      transform_config['output_dir'],
                                      constants.TRAINING_SUFFIX)
  schema_file_gcs_path = os.path.join(transform_dir, '/schema.pbtxt')
  features_config_file = transform_config['features_config']

  rand_id = str(uuid.uuid1())[:8]
  job_id = f'driblet-training-{rand_id}'
  job_dir = os.path.join(storage_bucket, training_config['job_dir'])

  args = {
      'transform_dir': transform_dir,
      'train_data': f'{transform_output_dir}/{train_file_suffix}*',
      'eval_data': f'{transform_output_dir}/{eval_file_suffix}*',
      'features_config_file': features_config_file,
      'job_dir': job_dir,
      'schema_file': schema_file_gcs_path,
      'model_name': training_config['model_name'],
      'train_steps': int(training_config['train_steps']),
      'eval_steps': int(training_config['eval_steps']),
      'train_batch_size': int(training_config['train_batch_size']),
      'eval_batch_size': int(training_config['eval_batch_size']),
      'estimator_type': training_config['estimator_type']
  }

  training_args = [
      f'--{key}="{value}"' for key, value in args.items() if value is not None
  ]

  _ = mlengine_operator.MLEngineTrainingOperator(
      project_id=project_id,
      job_id=job_id,
      package_uris=os.path.join(dag_dir, _TRAINING_PACKAGE),
      training_python_module=_TRAINING_MODULE,
      training_args=training_args,
      region=region,
      runtime_version=_RUNTIME_VERSION,
      python_version=_PYTHON_VERSION,
      job_dir=training_config['job_dir'],
      gcp_conn_id=gcp_conn_id,
      mode=mode,
      dag=dag,
      task_id=SUB_DAG_ID)

  return dag
