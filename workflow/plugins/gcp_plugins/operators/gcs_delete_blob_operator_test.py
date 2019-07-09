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
"""Test for gcs_delete_blob_operator."""

import unittest
import mock

from  import gcs_delete_blob_operator
from google.cloud import storage

_GCS_BUCKET = 'test_bucket'
_DIR_PREFIX = 'test_dir'
_FILE_NAME = _DIR_PREFIX + '/test_file'


class GCSDeleteBlobOperatorTest(unittest.TestCase):

  def setUp(self):
    super(GCSDeleteBlobOperatorTest, self).setUp()
    # Mock a GCS client that returns mock bucket with mock blob.
    self.mock_client = mock.create_autospec(storage.Client)
    self.mock_bucket = mock.create_autospec(storage.bucket.Bucket)
    self.mock_blob = mock.create_autospec(storage.blob.Blob)
    self.mock_blob.name = _FILE_NAME
    self.mock_client.get_bucket.return_true = self.mock_bucket
    self.mock_bucket.list_blobs.return_value = [self.mock_blob]
    self.context = mock.MagicMock()

  def test_delete_blob(self):
    """Test if storage.Blob.delete() method is called."""
    operator = gcs_delete_blob_operator.GCSDeleteBlobOperator(
        task_id='test_task_id',
        client=self.mock_client,
        gcs_bucket=_GCS_BUCKET,
        prefixes=[_DIR_PREFIX])
    operator.execute(self.context)
    bucket = operator._client.get_bucket(_GCS_BUCKET)
    bucket.get_blob(_FILE_NAME).delete.called_once()


if __name__ == '__main__':
  unittest.main()
