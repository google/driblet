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
"""Driblet Cloud Environment configuration setup module."""

import datetime
import json
import logging
import logging.config
import os
import re
import shlex
import subprocess
import time
import uuid

from googleapiclient import discovery
from googleapiclient import errors
import yaml
from google.auth.transport import requests as google_auth_requests
from google.cloud import bigquery
from google.cloud import exceptions
from google.cloud import storage
from google.oauth2 import service_account

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIGURATION = 'configuration.yaml'
_WAIT_FOR_COMLETION_SLEEP_SECONDS = 10
_SAVED_MODEL = 'saved_model'

# Custom logging setup
logging.config.fileConfig(os.path.join(_BASE_DIR, 'logging_config.ini'))
logger = logging.getLogger(__name__)


class SubprocessError(Exception):
  """A command exited with an error."""


def call_subprocess(cmd_args, output=True):
  """Calls subprocess with given command line arguments.

  Args:
    cmd_args: string of command line arguments
    output: If True, JSON output will be returned

  Returns:
    JSON result of subprocess output.
  Raises:
    SubprocessError.
  """
  try:
    if output:
      return json.loads(subprocess.check_output(shlex.split(cmd_args)))
    else:
      subprocess.check_output(shlex.split(cmd_args))
  except subprocess.CalledProcessError as error:
    if error.returncode == 0:
      logger.debug('Ignoring exit status "%s"', error.returncode)
      return
    raise SubprocessError(error)


def authenticate_gcloud():
  """Authenticates Google Cloud account."""
  cmd_account_info = 'gcloud info --format json'
  config = call_subprocess(cmd_account_info)
  if config['config']['account'] is None:
    cmd_login = 'gcloud auth login'
    call_subprocess(cmd_login, output=False)
  project_id = get_config_value('project')
  set_project_id(project_id)


def get_project_configs():
  """Gets Cloud project configurations.

  Returns:
    JSON object containing Cloud configuration.
  """
  cmd_config_info = 'gcloud config configurations list --format json'
  return call_subprocess(cmd_config_info)


def set_project_id(project_id):
  """Sets project ID for active environment.

  Args:
    project_id: Cloud project id.
  """
  cmd_set_project = 'gcloud config set project {}'.format(project_id)
  call_subprocess(cmd_set_project, output=False)


def get_config_value(config_key):
  """Retrieves value from Cloud configuration.

  Args:
    config_key: config key to retrieve value for.

  Returns:
    string value from config object.
  """
  for config in get_project_configs():
    value = config['properties']['core'][config_key]
    if value:
      return value
    else:
      logging.error('There is no active environment with given key "%s"',
                    config_key)


def get_auth_session(sa_name, service_scopes):
  """Creates AuthorizedSession for given service account.

  Args:
      sa_name: Service account name.
      service_scopes: A list of cloud account service scopes.

  Returns:
    AuthorizedSession.
  """
  service_account_file = '{}.json'.format(sa_name)
  credentials = service_account.Credentials.from_service_account_file(
      service_account_file, scopes=service_scopes)
  return google_auth_requests.AuthorizedSession(credentials)


def create_config_env(sa_name):
  """Creates and activates Cloud environment..

  Args:
      sa_name: Service account name.
  """
  authenticate_gcloud()
  env_exists = False
  env_name = '{}-env'.format(sa_name)
  for config in get_project_configs():
    if config['name'] == env_name:
      env_exists = True
      if not config['is_active']:
        cmd_config_activate = (
            'gcloud config configurations activate {}'.format(env_name))
        call_subprocess(cmd_config_activate, output=False)
        logger.info('"%s" environment is found. Setting it to active state.',
                    env_name)

  if not env_exists:
    cmd_config_create = ('gcloud config configurations create {} '
                         '--activate'.format(env_name))
    call_subprocess(cmd_config_create, output=False)
    logger.info(
        '"%s" environment not found. Creating and setting it to active state.',
        env_name)


def list_service_account_keys(sa_email):
  """Lists service account keys.

  Args:
    sa_email: The service account email whose keys to list.

  Returns:
    List of service account keys.
  """
  cmd_list_key = ('gcloud iam service-accounts keys list --iam-account={} '
                  '--format json'.format(sa_email))
  return call_subprocess(cmd_list_key)


def create_service_account_key(sa_name):
  """Create key for service account and save it to local directory.

  Args:
    sa_name: Cloud service account name.
  """
  sa = get_service_account(sa_name)
  sa_email = sa['email']
  key_list = list_service_account_keys(sa_email)
  sa_json = sa_email.split('@')[0] + '.json'
  if len(key_list) > 1:
    key_id = key_list[0]['name'].split('/')[-1]
    # Delete existing key in a service account.
    delete_service_account_key(key_id, sa_email)
  cmd_create_key = ('gcloud iam service-accounts keys create {} '
                    '--iam-account={}'.format(sa_json, sa_email))
  logger.info('Creating new service account key for "%s"', sa_email)
  call_subprocess(cmd_create_key, output=False)
  set_service_account_role(sa_email)


def delete_service_account_key(key_id, sa_email):
  """Deletes existing service account key.

  Args:
    key_id: Key id to delete.
    sa_email: The service account email whose keys to list.
  """
  cmd_remove_key = ('gcloud iam service-accounts keys delete {} -q '
                    '--iam-account={}'.format(key_id, sa_email))
  call_subprocess(cmd_remove_key, output=False)
  logger.info('Service account key "%s" has been deleted.', key_id)


def set_service_account_role(sa_email):
  """Assign editor role to the service account.

  Args:
    sa_email: The service account email whose keys to list.
  """
  project_id = get_config_value('project')
  cmd_assign = ('gcloud projects add-iam-policy-binding {}'
                ' --member=\'serviceAccount:{}\''
                ' --role roles/editor'.format(project_id, sa_email))
  call_subprocess(cmd_assign, output=False)
  logger.info('Assigned roles/editor to "%s"', sa_email)


def get_service_account(sa_name):
  """Lists existing service account given servive account name.

  Args:
    sa_name: Name of the Cloud service account.

  Returns:
    A dictionary of filtered service account.
  """
  cmd_sa_filter = ('gcloud iam service-accounts list --filter {} '
                   '--format json'.format(sa_name))
  service_acnt = call_subprocess(cmd_sa_filter)
  service_acnt = service_acnt[0] if len(service_acnt) else service_acnt
  return service_acnt


def create_service_account(sa_name):
  """Creates service account.

  Args:
    sa_name: Cloud service account name.
  """
  # TODO(azamat): switch to REST API.
  create_config_env(sa_name)
  cmd_sa_create = ('gcloud iam service-accounts create {} '
                   '--display-name {}'.format(sa_name, sa_name.upper()))
  call_subprocess(cmd_sa_create, output=False)
  logger.info(
      'Waiting for %d seconds for "create service account" job to complete.',
      _WAIT_FOR_COMLETION_SLEEP_SECONDS)
  time.sleep(_WAIT_FOR_COMLETION_SLEEP_SECONDS)
  logger.info('Service account "%s" has been created.', sa_name)


def filter_disabled_gcp_apis(project_id, auth_session, service_url,
                             required_apis):
  """Filters disabled API.

  Args:
    project_id: Cloud project id.
    auth_session: Instance of AuthorizedSession.
    service_url: Service URL.
    required_apis: A list of required Cloud APIs.

  Returns:
    List of disabled APIs.
  """
  service_url = '{}/{}/services?filter=state:ENABLED'.format(
      service_url, project_id)
  response = auth_session.get(service_url)
  response_content = json.loads(response.content)
  enabled_apis = [
      service['config']['name'] for service in response_content['services']
  ]
  disabled_apis = [api for api in required_apis if api not in enabled_apis]
  if disabled_apis:
    logging.info('Enabling APIs: "%s".', ', '.join(disabled_apis))
  return disabled_apis


def enable_gcp_apis(project_id, sa_name, service_url, service_scopes,
                    required_apis):
  """Enables required Cloud APIs for the project.

  Args:
    project_id: Cloud project id.
    sa_name: Service account name.
    service_url: Service URL.
    service_scopes: A list of cloud account service scopes.
    required_apis: A list of required Cloud APIs.
  """
  auth_session = get_auth_session(sa_name, service_scopes)
  disabled_apis = filter_disabled_gcp_apis(project_id, auth_session,
                                           service_url, required_apis)
  if disabled_apis:
    apis = json.dumps({'serviceIds': disabled_apis})
    response = auth_session.post(
        '{}/{}/services:batchEnable'.format(service_url, project_id), data=apis)
    # Wait till all required APIs are enabled.
    enabled = False
    while not enabled:
      disabled_apis = filter_disabled_gcp_apis(project_id, auth_session,
                                               service_url, required_apis)
      if not disabled_apis:
        enabled = True
      logger.info('Waiting for %d seconds for required APIs to be enabled.',
                  _WAIT_FOR_COMLETION_SLEEP_SECONDS)
      time.sleep(_WAIT_FOR_COMLETION_SLEEP_SECONDS)
    if response.status_code != 200:
      logger.error(
          'Error occurred while enabling APIs "%s". '
          'Error reason: "%s"', ', '.join(disabled_apis), response.reason)
  logger.info('All required APIs are enabled.')


class CloudStorageHelper(object):
  """Cloud Storage helper class."""

  def __init__(self, client):
    self._client = client

  def _blob_exists(self, bucket, blob):
    """Checks if given Cloud Storage blob exists.

    Args:
      bucket: Cloud Storage bucket object.
      blob: Cloud Storage blob object.

    Returns:
      True if blob exists, False otherwise.
    """
    blob = storage.Blob(bucket=bucket, name=blob.name)
    if blob.exists(self._client):
      return True
    else:
      return False

  def create_bucket(self, bucket_name, location, storage_class):
    """Creates bucket in the project.

    Args:
      bucket_name: Cloud Storage bucket name.
      location: Cloud Storage location.
      storage_class: Cloud Storage class.
    """
    try:
      bucket = storage.Bucket(self._client, name=bucket_name)
      bucket.location = location
      bucket.storage_class = storage_class
      bucket.create()
      logger.info('Bucket "%s" has been created.', bucket.name)
    except exceptions.Conflict:
      logger.info('Bucket "%s" already exists.', bucket.name)

  def remove_files(self, bucket_name, exclude_prefixes=None):
    """Removes files in a given bucket.

    Args:
      bucket_name: Cloud Storage bucket name.
      exclude_prefixes: List of directory prefixes for Cloud Storage bucket to
        exclude from deleting.
    """
    bucket = storage.Bucket(self._client, name=bucket_name)
    exclude_blobs = []
    if exclude_prefixes:
      for prefix in exclude_prefixes:
        for blob in bucket.list_blobs(prefix=prefix):
          exclude_blobs.append(blob.id)

    for blob in bucket.list_blobs():
      if self._blob_exists(bucket, blob) and blob.id not in exclude_blobs:
        blob.delete()
        logger.info('"%s" has been deleted.', blob.name)

  def upload_files(self, dest_bucket_name, source_file, dest_file):
    """Uploads given files from local directory to Cloud bucket.

    Args:
      dest_bucket_name: Cloud Storage destination bucket name.
      source_file: Local file name.
      dest_file: Destination file name.
    """
    bucket = storage.Bucket(self._client, name=dest_bucket_name)
    try:
      blob = bucket.blob(dest_file)
      blob.upload_from_filename(source_file)
      logger.info('Uploaded %s to %s', blob.name, bucket.name)
    except IOError as error:
      logger.error('Could not find files at "%s". Error: "%s"', source_file,
                   error)

  def copy_files(self, source_bucket_name, dest_bucket_name, prefix=None):
    """Copies files from one Cloud Storage bucket to another.

    Args:
      source_bucket_name: Source bucket name.
      dest_bucket_name: Destination bucket name.
      prefix: Prefix to filter bucket directories.
    """
    source_bucket = self._client.get_bucket(source_bucket_name)
    dest_bucket = self._client.get_bucket(dest_bucket_name)
    source_blobs = source_bucket.list_blobs(prefix=prefix)
    for blob in source_blobs:
      logger.info('Copying "%s"', blob.name)
      source_bucket.copy_blob(blob, dest_bucket)


class CloudComposerHelper(object):
  """Cloud Composer helper class."""

  def __init__(self, composer_service, composer_name, sa_name, location):
    self._composer_service = composer_service
    self._composer_name = composer_name
    self._sa_name = sa_name
    self._location = location

  def _env_exists(self, env_name):
    """Checks if composer environment exists.

    Args:
      env_name: Cloud Composer environment name.

    Returns:
      True if environment exists, False otherwise.
    """
    env_list = self.get_composer_envs(env_name)
    if env_list:
      if env_list['name'].split('/')[-1] == self._composer_name:
        return True
    return False

  def get_composer_envs(self, env_name):
    """Retrieves existing composer environments.

    Args:
      env_name: Cloud Composer environment name.

    Returns:
      Response body contains an instance of Composer environment.
    """
    try:
      request = (
          self._composer_service.projects().locations().environments().get(
              name=env_name))
      return request.execute()
    except errors.HttpError as error:
      logger.error('%s.', json.loads(error.content)['error']['message'])

  def _wait_composer_operation(self, operation):
    """Checks composer operation status.

    Args:
      operation: Request execute operation.

    Raises:
      RuntimeError if deployment completes with errors.
    """
    while True:
      deploy_status = (
          self._composer_service.projects().locations().operations().get(
              name=operation.get('name')).execute())
      if deploy_status.get('done'):
        logging.info('Composer operation successful.')
        break

      if deploy_status.get('error'):
        logging.error('Composer operation failed: %s', str(operation))
        raise RuntimeError('Failed to deploy composor: {}').format(
            deploy_status['error'])
      logger.info('Waiting for %d seconds for "%s" to complete.',
                  _WAIT_FOR_COMLETION_SLEEP_SECONDS, operation.get('name'))
      time.sleep(_WAIT_FOR_COMLETION_SLEEP_SECONDS)

  def set_env_variables(self, env_varialbes):
    """Sets environment varialbes to be used in Airflow workflow.

    Args:
      env_varialbes: Dictionary of environment variables to set for Airflow.
    """
    for key, value in env_varialbes.items():
      cmd_set = ('gcloud composer environments run {} --location {} '
                 'variables -- --set {} {}'.format(self._composer_name,
                                                   self._location, key, value))
      logger.info('Setting up env variables: "%s":"%s"', key, value)
      subprocess.check_call(shlex.split(cmd_set))

  def create_composer_env(self, project_id, env_name, machine_type):
    """Creates Cloud Composer environment.

    Args:
      project_id: Cloud project id.
      env_name: Cloud Composer environment name.
      machine_type: Machine type to run VM on.
    """
    # For details on composer REST api, refer to
    # https://cloud.google.com/composer/docs/reference/rest/v1/projects.locations.environments/create
    composer_zone = '{}-a'.format(self._location)
    request_body = {
        'name': env_name,
        'config': {
            'nodeConfig': {
                'location':
                    ('projects/{}/zones/{}'.format(project_id, composer_zone)),
                'machineType': ('projects/{}/zones/{}/machineTypes/{}'.format(
                    project_id, composer_zone, machine_type)),
                'diskSizeGb': 25  # Minimum is 20GB
            }
        }
    }
    parent = ('projects/{}/locations/{}'.format(project_id, self._location))
    request = (
        self._composer_service.projects().locations().environments().create(
            parent=parent, body=request_body))
    if not self._env_exists(env_name):
      try:
        operation = request.execute()
        self._wait_composer_operation(operation)
        logger.info('"%s" has been created.', self._composer_name)
      except errors.HttpError as error:
        message = json.loads(error.content)['error']['message']
        logger.error('"%s". Composer instance was not created.', message)
    else:
      logger.info('Composer environment "%s" exists.', self._composer_name)

  def update_composer_env(self, env_name, py_packages):
    """Updates Cloud Composer Environment.

    Args:
      env_name: Cloud Composer environment name.
      py_packages: Python packages to be instgalled to Composer environment.
    """
    request_body = {
        'name': env_name,
        'config': {
            'softwareConfig': {
                'pypiPackages': py_packages
            }
        }
    }
    request = (
        self._composer_service.projects().locations().environments().patch(
            name=env_name,
            body=request_body,
            updateMask='config.softwareConfig.pypiPackages'))
    try:
      operation = request.execute()
      self._wait_composer_operation(operation)
    except errors.HttpError as error:
      logging.error('"%s". Composer instance was not updated.', error)


class CloudMLEngineHelper(object):
  """Cloud ML Engine helper class."""

  def __init__(self, ml_service, project_id):
    self._ml_service = ml_service
    self._project_id = project_id

  def _model_exists(self, project_id, model_name):
    """Lists models in the project.

    Args:
      project_id: Cloud project id to list models for.
      model_name: Moddel name to check if exists.

    Returns:
      Dictionary of models.
    """
    models = self._ml_service.projects().models()
    try:
      request = models.list(parent='projects/{}'.format(project_id))
      response = request.execute()
      if response:
        for model in response['models']:
          if model['name'].rsplit('/', 1)[1] == model_name:
            return True
          else:
            return False
    except errors.HttpError as err:
      logger.error('%s', json.loads(err.content)['error']['message'])

  def _list_model_versions(self, model_name):
    """Lists existing model versions in the project.

    Args:
      model_name: Model name to list versions for.

    Returns:
      Dictionary of model versions.
    """
    versions = self._ml_service.projects().models().versions()
    request = versions.list(
        parent='projects/{}/models/{}'.format(self._project_id, model_name))
    try:
      return request.execute()
    except errors.HttpError as err:
      logger.error('%s', json.loads(err.content)['error']['message'])

  def create_model(self, model_name, model_region):
    """Creates model in ML Engines.

    Args:
      model_name: Model name to create.
      model_region: Region to deploy the model.
    """
    # For details on request body, refer to:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models/create
    if not self._model_exists(self._project_id, model_name):
      request_body = {
          'name': model_name,
          'regions': model_region,
          'description': 'Driblet model'
      }
      parent = 'projects/{}'.format(self._project_id)
      try:
        request = self._ml_service.projects().models().create(
            parent=parent, body=request_body)
        request.execute()
        logger.info('Model "%s" has been created.', model_name)
      except errors.HttpError as err:
        logger.error('"%s". Skipping model creation.',
                     json.loads(err.content)['error']['message'])
    else:
      logger.info('Model "%s" already exists.', model_name)

  def deploy_model(self, bucket_name, model_name, model_version,
                   runtime_version):
    """Deploys model on Cloud ML Engine.

    Args:
      bucket_name: Cloud Storage Bucket name that stores saved model.
      model_name: Model name to deploy.
      model_version: Model version.
      runtime_version: Runtime version.

    Raises:
      RuntimeError if deployment completes with errors.
    """
    # For etails on request body, refer to:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions/create
    model_version_exists = False
    model_versions_list = self._list_model_versions(model_name)

    if model_versions_list:
      for version in model_versions_list['versions']:
        if version['name'].rsplit('/', 1)[1] == model_version:
          model_version_exists = True

    if not model_version_exists:
      request_body = {
          'name': model_version,
          'deploymentUri': 'gs://{}/{}'.format(bucket_name, _SAVED_MODEL),
          'framework': 'TENSORFLOW',
          'runtimeVersion': runtime_version
      }
      parent = 'projects/{}/models/{}'.format(self._project_id, model_name)
      response = self._ml_service.projects().models().versions().create(
          parent=parent, body=request_body).execute()
      op_name = response['name']
      while True:
        deploy_status = (
            self._ml_service.projects().operations().get(
                name=op_name).execute())
        if deploy_status.get('done'):
          logger.info('Model "%s" with version "%s" deployed.', model_name,
                      model_version)
          break
        if deploy_status.get('error'):
          logging.error(deploy_status['error'])
          raise RuntimeError('Failed to deploy model for serving: {}').format(
              deploy_status['error'])
        logging.info(
            'Waiting for %d seconds for "%s" with "%s" version to be deployed.',
            _WAIT_FOR_COMLETION_SLEEP_SECONDS, model_name, model_version)
        time.sleep(_WAIT_FOR_COMLETION_SLEEP_SECONDS)
    else:
      logger.info('Model "%s" with version "%s" already exists.', model_name,
                  model_version)


class BigQueryHelper(object):
  """BigQuery helper class."""

  def __init__(self, client, dataset_name, table_name):
    self._client = client
    self._dataset_ref = self._client.dataset(dataset_name)
    self._table_ref = self._dataset_ref.table(table_name)

  def _dataset_exists(self, dataset_ref):
    """Checks if datast exists.

    Args:
      dataset_ref: BigQuery datasetReference object.

    Returns:
      True if exists, False otherwise.
    """
    try:
      self._client.get_dataset(dataset_ref)
      return True
    except exceptions.NotFound:
      logger.info('Dataset %s not found', dataset_ref.dataset_id)
      return False

  def _table_exists(self, table_ref):
    """Checks if table exists in a given dataset.

    Args:
      table_ref: BigQuery tableReference object.

    Returns:
      True if exists, False otherwise.
    """
    try:
      self._client.get_table(table_ref)
      return True
    except exceptions.NotFound:
      logger.info('Table "%s" not found', table_ref.table_id)
      return False

  def create_dataset_and_table(self, dataset_expiration, dataset_location):
    """Creates dataset and table in BigQuery.

    Args:
      dataset_expiration: Dataset expiration time in ms.
      dataset_location: Location for the dataset.
    """
    # Create dataset if doesn't exist.
    if not self._dataset_exists(self._dataset_ref):
      try:
        dataset = bigquery.Dataset(self._dataset_ref)
        dataset.location = dataset_location
        dataset.default_table_expiration_ms = dataset_expiration
        self._client.create_dataset(dataset)
        logger.info('"%s" has been created.', self._dataset_ref.dataset_id)
      except exceptions.Conflict:
        logger.info('Dataset already exists "%s".',
                    self._dataset_ref.dataset_id)

    # Create table if doesn't exist.
    if not self._table_exists(self._table_ref):
      try:
        table = bigquery.Table(self._table_ref)
        self._client.create_table(table)
        logger.info('"%s" has been created.', self._table_ref.table_id)
      except exceptions.Conflict:
        logger.info('Table already exists "%s".', self._table_ref.table_id)

  def load_data(self, gcs_file_path):
    """Loads sample data to sample table.

    Args:
      gcs_file_path: Cloud storage path for CSV.
    """
    destination_table = self._client.get_table(self._table_ref)
    if destination_table.num_rows < 1:
      job_config = bigquery.LoadJobConfig()
      job_config.autodetect = True
      job_config.skip_leading_rows = 1
      job_config.source_format = bigquery.SourceFormat.CSV
      load_job = self._client.load_table_from_uri(
          gcs_file_path, self._table_ref, job_config=job_config)
      load_job.result()  # Waits for table load to complete.
      logger.info('Finished loading data. Job id: "%s"', load_job.job_id)
    else:
      logger.info('Required data already exists in "%s".',
                  destination_table.table_id)


def construct_file_paths(local_config):
  """Constructs source and destination file paths to upload to Cloud Storage.

  Args:
    local_config: Local configuration variables.

  Returns:
    source_dest_paths: A list containing source and destination file paths.
  """
  paths = [os.path.expanduser(path) for path in local_config.values()]
  source_dest_paths = []

  for path in paths:
    if os.path.isfile(path):
      _, pardir = os.path.split(os.path.dirname(path))
      dest_file = os.path.join(pardir, os.path.basename(path))
      source_dest_paths.append((path, dest_file))
    for root, _, files in os.walk(path):
      # Find exported model directory in a root path
      model_rootdir = ''.join(re.findall(r'\d{10}', root))
      for filename in files:
        # Exclude test files from uploading to Airflow dag's folder
        if 'test' not in filename:
          source_file = os.path.join(root, filename)
          dest_file = os.path.join(*source_file.split('/')[1:])
        # If saved model exists, construct Cloud Storage destination path which
        # is used to deploy model on AI Platform
        if model_rootdir:
          _, model_sub_dir = os.path.split(os.path.dirname(source_file))
          source_basename = os.path.basename(source_file)
          if model_sub_dir == model_rootdir:
            dest_file = os.path.join(_SAVED_MODEL, source_basename)
          else:
            dest_file = os.path.join(_SAVED_MODEL, model_sub_dir,
                                     source_basename)
        source_dest_paths.append((source_file, dest_file))
  return source_dest_paths


def main():
  # Load local configuration file.
  with open(_CONFIGURATION, 'r') as f:
    try:
      config = yaml.safe_load(f)
    except yaml.YAMLError as error:
      logger.error(error)

  # Set up Cloud environment.
  sa_name = '{}-sa'.format(config['env_prefix'])
  if not get_service_account(sa_name):
    create_service_account(sa_name)
  create_service_account_key(sa_name)
  project_id = get_config_value('project')
  enable_gcp_apis(project_id, sa_name, config['service_url'],
                  config['service_scopes'], config['required_apis'])

  # Setup Cloud Storage environment.
  gcs_client = storage.Client(project=project_id)
  gcs_helper = CloudStorageHelper(client=gcs_client)
  # Obtain local and destination file paths to upload to newly created bucket.
  source_dest_paths = construct_file_paths(config['local_config'])
  bucket_name = '{}_bucket_{}'.format(config['env_prefix'], str(uuid.uuid4()))
  gcs_helper.create_bucket(bucket_name, config['location'],
                           config['storage_class'])
  # Upload local files to Cloud Storage bucket.
  for source_file, dest_file in source_dest_paths:
    gcs_helper.upload_files(bucket_name, source_file, dest_file)

  # Setup AI Platform environment.
  ml_service = discovery.build('ml', 'v1', cache_discovery=False)
  ml_helper = CloudMLEngineHelper(ml_service=ml_service, project_id=project_id)
  ml_helper.create_model(config['model_name'], config['location'])
  ml_helper.deploy_model(bucket_name, config['model_name'],
                         config['model_version'], config['runtime_version'])

  # Setup BigQuery environment.
  bq_client = bigquery.Client(project=project_id)
  table_name = '{}_{}'.format(config['table_name'],
                              datetime.datetime.now().strftime('%Y%m%d'))
  bq_helper = BigQueryHelper(
      client=bq_client,
      dataset_name=config['dataset_name'],
      table_name=table_name)
  bq_helper.create_dataset_and_table(config['dataset_expiration'],
                                     config['location'])
  sample_data = config['local_config']['sample_data']
  sample_data_dir = os.path.split(os.path.dirname(sample_data))[-1]
  sample_data_file = os.path.join(sample_data_dir,
                                  os.path.basename(sample_data))
  sample_data_blob = 'gs://{}/{}'.format(bucket_name, sample_data_file)
  bq_helper.load_data(sample_data_blob)

  # Setup Cloud Composer environment.
  composer_service = discovery.build('composer', 'v1', cache_discovery=False)
  composer_name = '{}-composer'.format(config['env_prefix'])
  composer_env_name = ('projects/{}/locations/{}/environments/{}'.format(
      project_id, config['location'], composer_name))
  composer_helper = CloudComposerHelper(
      composer_service=composer_service,
      composer_name=composer_name,
      sa_name=sa_name,
      location=config['location'])
  composer_env_variables = {
      'project_id': project_id,
      'bucket_name': bucket_name,
      'bq_dataset': config['dataset_name'],
      'bq_input_table': config['table_name'],
      'bq_output_table': '{}_output'.format(project_id.replace('-', '_')),
      'model_name': config['model_name'],
      'model_version': config['model_version'],
      'location': config['location'],
      'region': config['region'],
      'dataset_expiration': config['dataset_expiration']
  }
  composer_helper.create_composer_env(project_id, composer_env_name,
                                      config['machine_type'])
  # Update composer environment with required python packages.
  composer_helper.update_composer_env(composer_env_name,
                                      config['required_py_packages'])
  # Set Airflow environment variables to be used in DAG.
  composer_helper.set_env_variables(composer_env_variables)
  composer_env = composer_helper.get_composer_envs(composer_env_name)
  # Storage bucket is uniquely generated for Composer DAG. In order to move DAG
  # files, Storage bucket name is extracted from Composer environment.
  destination_dag_path = composer_env['config']['dagGcsPrefix'].split('/')[2]
  # Copy DAG files and plugins manage Airflow workflow.
  # Python modules in `plugins` directory will be copied to
  # `gs://<composer_dag_path/plugins>`. Same applies for `dags` directory.
  for prefix in ['plugins', 'dags']:
    gcs_helper.copy_files(bucket_name, destination_dag_path, prefix=prefix)

  # Delete files in GCS bucket after they were moved to ML Engine and Composer.
  gcs_helper.remove_files(bucket_name, exclude_prefixes=['transformer'])


if __name__ == '__main__':
  main()
