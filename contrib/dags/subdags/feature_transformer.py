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
"""Airflow DAG to run feature transform on DataFlow.

The DataFlow operator perform the necessaray normalization and transformations,
then write the result data to Cloud Storage as tfrecord files.
"""
import datetime
import os
from airflow import configuration
from airflow import models

from driblet.contrib.plugins.operators import dataflow_py3_operator
from driblet.dags import constants
from driblet.dags import utils

# Name of the SubDAG.
SUB_DAG_ID = 'feature-transform'


def create_dag(parent_dag_id: str,
               train_mode: bool,
               data_source: str = 'bigquery') -> models.DAG:
  """Creates DAG to run feature transform on DataFlow.

  Args:
    parent_dag_id: Name of the parent DAG.
    train_mode: If set true, pipeline will run in train mode to transform train,
      eval and predict data. Otherwise, it transforms only predict data. Should
      be set False in prediction mode.
    data_source: Dataset source for transformation. Should be one either `csv`
      or `bigquery`.

  Returns:
      airflow.models.DAG: The DAG object.
  """
  base_config = utils.get_airflow_variable_as_dict(constants.BASE_CONFIG)
  gcp_config = utils.get_airflow_variable_as_dict(constants.GCP_CONFIG)
  dataset_split_config = utils.get_airflow_variable_as_dict(
      constants.DATASET_SPLIT_CONFIG)
  transform_config = utils.get_airflow_variable_as_dict(
      constants.TRANSFORM_CONFIG)

  project_id = gcp_config['project_id']
  storage_bucket = gcp_config['storage_bucket']

  # By convention, a SubDAG's name should be prefixed by its parent and a dot.
  dag_id = f'{parent_dag_id}.{SUB_DAG_ID}'
  dag_schedule_interval = base_config['schedule_interval']
  dag_retries = constants.DAG_RETRIES
  dag_retry_delay = constants.DAG_RETRY_DELAY
  dag = utils.create_dag(dag_id, dag_schedule_interval, dag_retries,
                         dag_retry_delay)
  dag_dir = configuration.get('core', 'dags_folder')
  transformer_py = os.path.join(dag_dir, 'tasks/transformer',
                                'feature_transformer.py')
  transform_dir = os.path.join(storage_bucket,
                               transform_config['transform_dir'])
  features_config = os.path.join(storage_bucket,
                                 transform_config['features_config'])
  output_dir = transform_config['output_dir']
  all_data = dataset_split_config['input_table']
  options = {
      'project_id': project_id,
      'data_source': data_source,
      'mode_train': 'True' if train_mode else 'False',
      'transform_dir': transform_dir,
      'features_config': features_config
  }

  if train_mode:
    suffix = constants.TRAINING_SUFFIX
    options['all_data'] = f'{all_data}_{suffix}'
    options['train_data'] = dataset_split_config['train_dest_table']
    options['eval_data'] = dataset_split_config['eval_dest_table']
    options['predict_data'] = dataset_split_config['test_dest_table']
    options['output_dir'] = os.path.join(storage_bucket, output_dir, suffix, '')
  else:
    suffix = constants.PREDICTION_SUFFIX
    today = datetime.datetime.now().strftime('%Y%m%d')
    options['predict_data'] = f'{all_data}_{suffix}_{today}'
    options['output_dir'] = os.path.join(storage_bucket, output_dir, suffix, '')

  # dag only contains one operator
  _ = dataflow_py3_operator.DataFlowPythonOperator(
      task_id='feature_transform',
      py_file=transformer_py,
      options=options,
      dataflow_default_options={'project': project_id},
      dag=dag)

  return dag
