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

"""Tests for driblet.cloud_env_setup."""

import argparse
import unittest
import mock
import parameterized
from gps_building_blocks.cloud.utils import cloud_api
from gps_building_blocks.cloud.utils import cloud_composer
from gps_building_blocks.cloud.utils import cloud_storage
from driblet import cloud_env_setup


class CloudEnvSetupTest(unittest.TestCase):

  def setUp(self):
    super(CloudEnvSetupTest, self).setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_parse_args = mock.patch.object(
        cloud_env_setup, 'parse_arguments', autospec=True).start()
    self.service_account_name = 'my-service@my-project.iam.gserviceaccount.com'
    self.composer_env_name = 'test-env'
    self.project_id = 'project_id'
    self.pypi_packages = {'test-package': '2.0'}
    self.modeling_platform = 'TensorFlow'
    self.mock_parse_args.return_value = argparse.Namespace(
        project_id=self.project_id,
        service_account_name=self.service_account_name,
        modeling_platform=self.modeling_platform,
        composer_env_name=cloud_env_setup._COMPOSER_ENV_NAME,
        dags_folder=cloud_env_setup._DAGS_FOLDER,
        saved_model_folder=cloud_env_setup._SAVED_MODEL_FOLDER,
        gcs_path_prefix=cloud_env_setup._GCS_PATH_PREFIX,
        config_file=cloud_env_setup._FEATURES_CONFIG_FILE)

    # Setup cloud_api mocks.
    self.mock_cloud_api_utils = mock.patch.object(
        cloud_api, 'CloudApiUtils', autospec=True).start()
    self.mock_cloud_composer_utils = mock.patch.object(
        cloud_composer, 'CloudComposerUtils', autospec=True).start()
    self.mock_cloud_storage_utils = mock.patch.object(
        cloud_storage, 'CloudStorageUtils', autospec=True).start()

  def test_enable_apis_enables_cloud_apis(self):
    mock_enable_apis = self.mock_cloud_api_utils.return_value.enable_apis

    cloud_env_setup.enable_apis(self.project_id, self.service_account_name)

    self.mock_cloud_api_utils.assert_called_once_with(
        project_id=self.project_id,
        service_account_name=self.service_account_name)
    mock_enable_apis.assert_called_once_with(
        cloud_env_setup._APIS_TO_BE_ENABLED)

  def test_install_pypi_packages_raises_value_error(self):
    modeling_platform = 'CustomPlatform'

    with self.assertRaises(ValueError):
      cloud_env_setup.install_pypi_packages(self.mock_cloud_composer_utils,
                                            self.composer_env_name,
                                            modeling_platform,
                                            self.pypi_packages)

  @parameterized.parameterized.expand([['TensorFlow'], ['AutoMl']])
  def test_install_pypi_packages_installs_python_packages(
      self, modeling_platform):
    mock_install_packages = (
        self.mock_cloud_composer_utils.install_python_packages)
    expected_pypi_packages = self.pypi_packages.copy()

    if modeling_platform == 'TensorFlow':
      expected_pypi_packages.update(cloud_env_setup._TENSORFLOW_MODULES)

    cloud_env_setup.install_pypi_packages(self.mock_cloud_composer_utils,
                                          self.composer_env_name,
                                          modeling_platform, self.pypi_packages)

    mock_install_packages.assert_called_once_with(self.composer_env_name,
                                                  expected_pypi_packages)

  def test_copy_dags_to_gcs_copies_files_and_dirs_to_gcs(self):
    mock_upload_directory = (
        self.mock_cloud_storage_utils.upload_directory_to_url)
    dags_folder = 'local_dags_folder'
    dags_folder_url = 'gcs_dags_folder/subfolder'
    self.mock_cloud_composer_utils.get_dags_folder.return_value = (
        dags_folder_url)

    cloud_env_setup.copy_dags_to_gcs(self.mock_cloud_composer_utils,
                                     self.mock_cloud_storage_utils,
                                     self.composer_env_name, dags_folder)

    mock_upload_directory.assert_called_once_with(dags_folder,
                                                  'gcs_dags_folder/dags')

  def test_copy_model_to_gcs_copies_saved_model_and_config_file_to_gcs(self):
    mock_upload_directory = (
        self.mock_cloud_storage_utils.upload_directory_to_url)
    mock_upload_file_to_url = self.mock_cloud_storage_utils.upload_file_to_url

    cloud_env_setup.copy_model_to_gcs(self.mock_cloud_storage_utils,
                                      'test_path_prefix', 'src/saved_model',
                                      'config.cfg')

    self.assertEqual(mock_upload_directory.call_count, 1)

    self.assertEqual(mock_upload_file_to_url.call_count, 1)


if __name__ == '__main__':
  unittest.main()
