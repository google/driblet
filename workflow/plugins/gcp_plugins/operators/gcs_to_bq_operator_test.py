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
"""Test for gcs_to_bigquery_operator."""

import unittest
import mock

from  import gcs_to_bq_operator
from google.cloud import bigquery
from google.cloud import storage

_DATASET_ID = 'test_dataset'
_TABLE_ID = 'test_table'
_LOCATION = 'test_location'
_DIR_PREFIX = 'test_dir'
_EXCLUDE_PREFIX = 'test_exclude'
_FILE_NAME = _DIR_PREFIX + '/test_file'


class GCStoBQOperatorTest(unittest.TestCase):

  def setUp(self):
    super(GCStoBQOperatorTest, self).setUp()
    # Mock a GCS client that returns mock bucket with mock blob.
    self.mock_gcs_client = mock.create_autospec(storage.Client)
    self.mock_bucket = mock.create_autospec(storage.bucket.Bucket)
    self.mock_bucket.name = 'test_bucket'
    self.mock_gcs_client.get_bucket.return_value = self.mock_bucket
    self.mock_blob = mock.create_autospec(storage.blob.Blob)
    self.mock_blob.name = _FILE_NAME
    self.mock_blob.size = 100
    self.mock_bucket.list_blobs.return_value = [self.mock_blob]
    self.context = mock.MagicMock()
    # Mock BigQuery client that returns mock dataset with mock table reference.
    self.mock_bq_client = mock.create_autospec(bigquery.Client)
    self.mock_dataset_ref = self.mock_bq_client.dataset(_DATASET_ID)
    self.mock_table_ref = self.mock_dataset_ref.table(_TABLE_ID)
    self.mock_bq_client.dataset.return_value = self.mock_dataset_ref
    self.job_config = bigquery.LoadJobConfig()
    self.operator = gcs_to_bq_operator.GCStoBQOperator(
        task_id='test_task_id',
        bq_client=self.mock_bq_client,
        gcs_client=self.mock_gcs_client,
        job_config=self.job_config,
        dataset_id=_DATASET_ID,
        table_id=_TABLE_ID,
        gcs_bucket=self.mock_bucket.name,
        gcs_location=_LOCATION,
        exclude_prefix=_EXCLUDE_PREFIX,
        dir_prefix=_DIR_PREFIX)

  def test_filter_blobs(self):
    """Test for filtering Storage blobs."""
    expected_output = ['gs://{}/{}'.format(self.mock_bucket.name, _FILE_NAME)]
    blobs = self.operator.filter_blobs(_EXCLUDE_PREFIX, _DIR_PREFIX)
    self.assertEqual(blobs, expected_output)

  def test_load_table(self):
    """Test for loading data to BigQuery table."""
    self.operator.execute(self.context)
    blobs = self.operator.filter_blobs(_EXCLUDE_PREFIX, _DIR_PREFIX)
    self.mock_bq_client.load_table_from_uri.assert_called_once_with(
        blobs[0],
        self.mock_table_ref,
        location=_LOCATION,
        job_config=self.job_config)


if __name__ == '__main__':
  unittest.main()
