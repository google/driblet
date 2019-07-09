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
"""Custom Operator to delete files in Cloud Storage bucket."""

import logging

from airflow import models

logger = logging.getLogger(__name__)


class GCSDeleteBlobOperator(models.BaseOperator):
  """Operator to delete files in a given Cloud Storage bucket."""

  def __init__(self, client, gcs_bucket, prefixes, *args, **kwargs):
    super(GCSDeleteBlobOperator, self).__init__(*args, **kwargs)
    self._client = client
    self._gcs_bucket = gcs_bucket
    self._prefixes = prefixes

  def execute(self, context):
    """Execute operator.

    This method is invoked by Airflow to delete files in Cloud Storage bucket.

    Args:
      context: Airflow context that contains references to related objects to
        the task instance.
    """
    bucket = self._client.get_bucket(self._gcs_bucket)
    for prefix in self._prefixes:
      for blob in bucket.list_blobs(prefix=prefix):
        if blob:
          blob.delete()
        logging.info('Deleted "%s" blob', blob.name)
