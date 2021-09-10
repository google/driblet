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
"""Contains all the common constants related to DAGs.

Any constant value which needs to be shared between DAGs or SubDAGs should be
added here.
"""
import enum


class PipelineMode(enum.Enum):
  """Pipeline running mode: either training or prediction pipieline."""
  TRAIN = 'train'
  PREDICT = 'predict'


class FeatureGenerator(enum.Enum):
  """Feature generator types.

  1. `custom` generates features from SQL script supplied by user. By default,
  baseline feature generator SQL is provided by FEATURES_SQL_PATH defined below.

  2. `mlwp` generates features from based on
  https://github.com/google/gps_building_blocks/tree/master/py/gps_building_blocks/ml/data_prep/ml_windowing_pipeline.
  The feature generation logic could be different according to PipelineMode.
  Which feature generator to use is defined in FEATURES_CONFIG dict on Airflow
  UI.
  """
  CUSTOM = 'custom'
  MLWP = 'mlwp'


# Airflow environment constant variable.
# ------------------------------------------------------------------------------
AIRFLOW_ENV = 'AIRFLOW_HOME'
LOCAL_TIMEZONE = 'Asia/Tokyo'

# Airflow DAG schedule and retry configs.
DAG_RETRIES = 0
DAG_RETRY_DELAY = 3  # In minutes

BASE_CONFIG = 'base_config'
GCP_CONFIG = 'gcp_config'
MWP_CONFIG = 'mwp_config'
CUSTOM_QUERY_CONFIG = 'custom_query_config'
PREDICTION_PIPELINE_CONFIG = 'prediction_pipeline_config'

# Custom features SQL script used to generate features for training and
# prediction pipelines. NOTE: This script is only used when
# `FeatureGenerator.CUSTOM` is passed to feature_pipeline.
FEATURES_SQL_PATH = 'queries/features.sql'
# SQL script used for batch prediction.
PREDICTION_SQL_PATH = 'queries/prediction.sql'

# Airflow environment constant variable.
# ------------------------------------------------------------------------------
# Date format used across pipelines.
DATE_FORMAT = '%Y%m%d_%H%M%S'

# Pipeline DAG ids
MWP_SESSION_DAG_ID = 'session_pipeline'
MWP_FEATURE_DAG_ID = 'feature_pipeline'
TRAINING_DAG_ID = 'training_pipeline'
MODEL_DEPLOY_DAG_ID = 'model_deploy_pipeline'
PREDICTION_DAG_ID = 'prediction_pipeline'

# ML Windowing Pipeline configs
# ------------------------------------------------------------------------------
# These variable key are set in Airflow Web UI and used by dags/subdags in this
# module.

# MLWP template paths
MWP_SESSION_TMPL = 'mwp-templates/user-session/UserSessionPipeline'
MWP_VISUALIZATION_TMPL = 'mwp-templates/data-visualization/DataVisualizationPipeline'
MWP_WINDOW_TMPL = 'mwp-templates/sliding-window/SlidingWindowPipeline'
MWP_FEATURE_TMPL = 'mwp-templates/generate-features/GenerateFeaturesPipeline'

# MLWP .avro file output paths
MWP_SESSION_OUTPUT_DIR = 'mwp-datasets/session/'
MWP_WINDOW_OUTPUT_DIR = 'mwp-datasets/feature/'

# MLWP BigQuery output tables
MWP_FACTS_TABLE = 'facts'
MWP_INSTANCE_TABLE = 'instances'
MWP_FEATURES_TABLE_PREFIX = 'features'

# Airflow Variable Keys defined on Airflow UI
MWP_SESSION_CONFIG = 'mwp_session_config'
MWP_TIME_CONFIG = 'mwp_time_config'
MWP_FEATURE_CONFIG = 'mwp_feature_config'

# Cloud storage dataset output folders
PREDICTION_SUFFIX = 'prediction'
TRAINING_SUFFIX = 'training'

# Feature transformation/training/model deployment pipeline configs
# ------------------------------------------------------------------------------
# These variables are used to train TensorFlow model, which include
# configurations for:
#  * splitting dataset into train/eval/test
#  * transforming datasets into `.tfrecord` format & normalize data
#  * constants for file/directory prefixes/suffixes

# Airflow Variable Keys defined on Airflow UI
DATASET_SPLIT_CONFIG = 'dataset_split_config'
TRAINING_CONFIG = 'training_config'
MODEL_DEPLOY_CONFIG = 'model_deploy_config'

# Feature transform config file path
TRANSFORM_CONFIG = 'configs/features_config.cfg'

# Feature transform fodler names
TRANSFORM_OUTPUT_DIR = 'transformed_data'
TRANSFORM_MODEL_DIR = 'transform_model'
# Feature transformation output TFrecord file suffixes
TRAIN_TFRECORD_SUFFIX = 'train'
EVAL_TFRECORD_SUFFIX = 'eval'
TEST_TFRECORD_SUFFIX = 'test'
PREDICT_TFRECORD_PREFIX = 'predict'

# Prediction and activation pipeline configs
# ------------------------------------------------------------------------------
# Airflow Variable Keys defined on Airflow UI
PREDICTION_CONFIG = 'prediction_config'
STORAGE_TO_ADS_CONFIG = 'storage_to_ads_config'
