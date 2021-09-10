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

"""Cloud Environment setup module."""

import argparse
import enum
import logging
import os
from typing import Dict
import uuid

from gps_building_blocks.cloud.utils import cloud_api
from gps_building_blocks.cloud.utils import cloud_composer
from gps_building_blocks.cloud.utils import cloud_storage

_COMPOSER_ENV_NAME = 'driblet-env'

# Required Cloud APIs to be enabled.
_APIS_TO_BE_ENABLED = [
    'bigquery-json.googleapis.com',
    'cloudapis.googleapis.com',
    'cloudresourcemanager.googleapis.com',
    'composer.googleapis.com',
    'dataflow.googleapis.com',
    'googleads.googleapis.com',
    'ml.googleapis.com',
    'monitoring.googleapis.com',
    'storage-api.googleapis.com',
    'storage-component.googleapis.com',
]
# Required Python packages.
_COMPOSER_PYPI_PACKAGES = {
    'googleads': '',
    'frozendict': '',
    'gps-building-blocks': ''
}
_TENSORFLOW_MODULES = {
    'apache-beam': '[gcp]==2.15.0',
    'tensorflow': '==1.14.0',
    'tensorflow-data-validation': '==0.13.0'
}
# Composer environment variables.
_COMPOSER_ENV_VARIABLES = {'PYTHONPATH': '/home/airflow/gcs/plugins'}

# Local folder names.
_DAGS_FOLDER = 'src/'
_SAVED_MODEL_FOLDER = 'saved_model'
_FEATURES_CONFIG_FILE = 'configs/features_config.cfg'
_GCS_PATH_PREFIX = 'gs://driblet-bucket'

# Set logging level.
logging.getLogger().setLevel(logging.INFO)


def parse_arguments() -> argparse.Namespace:
  """Initialize command line parser using argparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_id', help='GCP project id.', required=True)
  parser.add_argument(
      '--composer_env_name',
      help='GCP cloud composer environment name.',
      default=_COMPOSER_ENV_NAME,
      required=False)
  parser.add_argument(
      '--dags_folder',
      help='Path of the DAGs folder.',
      default=_DAGS_FOLDER,
      required=False)
  parser.add_argument(
      '--modeling_platform',
      help='Model platform. Should be one of {AutoMl, BigQueryMl, TensorFlow}.',
      required=True)
  parser.add_argument(
      '--saved_model_folder',
      help='Local folder containing saved Tensorflow model.',
      default=_SAVED_MODEL_FOLDER,
      required=False)
  parser.add_argument(
      '--gcs_path_prefix',
      help='Cloud Storage path prefix to upload saved_model and config file.',
      default=_GCS_PATH_PREFIX,
      required=False)
  parser.add_argument(
      '--config_file',
      help='Local path to features config file.',
      default=_FEATURES_CONFIG_FILE,
      required=False)
  parser.add_argument(
      '--service_account_name',
      help='Service account name. Ex: my-service@my-project.iam.gserviceaccount.com',
      required=True)

  return parser.parse_args()


class ModelingPlatform(enum.Enum):
  """Indicates the type of modeling platform used for the pipeline."""
  AUTOML = 'AUTOML'
  BIGQUERYML = 'BIGQUERYML'
  TENSORFLOW = 'TENSORFLOW'


def enable_apis(project_id: str, service_account_name: str) -> None:
  """Enables required Cloud APIs.

  Args:
    project_id: Google Cloud Project Id. Ex: my-project-id.
    service_account_name: Service account name. Ex:
      example@my-project-id.iam.gserviceaccount.com.
  """

  cloud_api_utils = cloud_api.CloudApiUtils(
      project_id=project_id, service_account_name=service_account_name)
  cloud_api_utils.enable_apis(_APIS_TO_BE_ENABLED)


def install_pypi_packages(
    cloud_composer_utils: cloud_composer.CloudComposerUtils,
    composer_env_name: str, modeling_platform: str,
    pypi_packages: Dict[str, str]) -> None:
  """Installs PyPi packages to Composer environment.

  Args:
    cloud_composer_utils: Instance of CloudComposerUtils.
    composer_env_name: Name of the Composer environment. Ex: my-env.
    modeling_platform: Which platform model is built on. Should be one of
      {AutoMl, BigQueryMl, TensorFlow}.
    pypi_packages: List of Python packages to install.

  Raises:
    ValueError: If provided modeling platform is not supported.
  """
  platform = modeling_platform.upper()
  supported_platforms = [platform.value for platform in ModelingPlatform]
  if platform not in supported_platforms:
    raise ValueError('%s is not supported. Choose one of {%s}.' %
                     (platform, ', '.join(supported_platforms)))
  if platform == ModelingPlatform.TENSORFLOW.name:
    pypi_packages = dict(pypi_packages, **_TENSORFLOW_MODULES)
  cloud_composer_utils.install_python_packages(composer_env_name, pypi_packages)


def copy_dags_to_gcs(cloud_composer_utils: cloud_composer.CloudComposerUtils,
                     cloud_storage_utils: cloud_storage.CloudStorageUtils,
                     composer_env_name: str, dags_folder: str) -> None:
  """Copies DAG modules to Google Cloud Storage (deploys Airflow modules).

  Args:
    cloud_composer_utils: Instance of CloudComposerUtils.
    cloud_storage_utils: Instance of CloudStorageUtils.
    composer_env_name: Name of the Composer environment. Ex: my-env.
    dags_folder: Path of the DAGs folder.
  """
  # Copy DAGs and dependencies.
  dags_folder_url = cloud_composer_utils.get_dags_folder(composer_env_name)
  dags_bucket = os.path.dirname(dags_folder_url)
  cloud_storage_utils.upload_directory_to_url(dags_folder, dags_bucket)


def copy_model_to_gcs(cloud_storage_utils: cloud_storage.CloudStorageUtils,
                      gcs_path_prefix: str, saved_model_folder: str,
                      config_file: str) -> None:
  """Copies saved model files to Google Cloud Storage.

  This step is required to deploy saved model to AI Platform.

  Args:
    cloud_storage_utils: Instance of CloudStorageUtils.
    gcs_path_prefix: Cloud Storage path prefix to upload saved model files.
    saved_model_folder: Local folder containing saved Tensorflow model.
    config_file: Local path to features config file.
  """
  unique_gcs_path = '{}-{}'.format(gcs_path_prefix, str(uuid.uuid4()))
  gcs_model_path = os.path.join(unique_gcs_path, saved_model_folder)
  cloud_storage_utils.upload_directory_to_url(saved_model_folder,
                                              gcs_model_path)
  # Copy features_config file to Cloud Storage.
  gcs_features_config_path = os.path.join(unique_gcs_path, config_file)
  cloud_storage_utils.upload_file_to_url(config_file, gcs_features_config_path)


def main():
  args = parse_arguments()
  cloud_composer_utils = cloud_composer.CloudComposerUtils(
      project_id=args.project_id,
      service_account_name=args.service_account_name)
  cloud_storage_utils = cloud_storage.CloudStorageUtils(
      project_id=args.project_id,
      service_account_name=args.service_account_name)

  enable_apis(args.project_id, args.service_account_name)

  # Create Composer environment
  cloud_composer_utils.create_environment(args.composer_env_name)
  # Set Composer Environment variables
  cloud_composer_utils.set_environment_variables(args.composer_env_name,
                                                 _COMPOSER_ENV_VARIABLES)
  install_pypi_packages(cloud_composer_utils, args.composer_env_name,
                        args.modeling_platform, _COMPOSER_PYPI_PACKAGES)
  copy_dags_to_gcs(cloud_composer_utils, cloud_storage_utils,
                   args.composer_env_name, args.dags_folder)
  if (args.modeling_platform.upper() == ModelingPlatform.TENSORFLOW.name or
      os.path.isdir(args.saved_model_folder)):
    copy_model_to_gcs(cloud_composer_utils, args.gcs_path_prefix,
                      args.saved_model_folder, args.config_file)


if __name__ == '__main__':
  main()
