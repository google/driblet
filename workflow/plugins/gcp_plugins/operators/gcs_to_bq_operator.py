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
"""Custom Operator to export Cloud Storage file to BigQuery table."""

import logging

from airflow import models

logger = logging.getLogger(__name__)


class GCStoBQOperator(models.BaseOperator):
  """Operator to export files fro Cloud Storage to BigQuery table."""

  def __init__(self, bq_client, gcs_client, job_config, dataset_id, table_id,
               gcs_bucket, gcs_location, exclude_prefix, dir_prefix, *args,
               **kwargs):
    super(GCStoBQOperator, self).__init__(*args, **kwargs)
    self._bq_client = bq_client
    self._gcs_client = gcs_client
    self._job_config = job_config
    self._dataset_id = dataset_id
    self._table_id = table_id
    self._gcs_bucket = gcs_bucket
    self._gcs_location = gcs_location
    self._exclude_prefix = exclude_prefix
    self._dir_prefix = dir_prefix

  def filter_blobs(self, exclude_prefix, dir_prefix):
    """Retrieve blobs in Cloud Storage bucket.

    Args:
      exclude_prefix: File prefix to exclude files containing it.
      dir_prefix: Directory prefix to filter directories.

    Returns:
      gcs_blobs: List of blob pathes in the bucket.
    """
    bucket = self._gcs_client.get_bucket(self._gcs_bucket)
    gcs_blobs = []
    for blob in bucket.list_blobs(prefix=dir_prefix):
      if exclude_prefix not in blob.name and blob.size:
        gcs_blobs.append('gs://{}/{}'.format(self._gcs_bucket, blob.name))
    return gcs_blobs

  def execute(self, context):
    """Execute operator.

    This method is invoked by Airflow to export files from Cloud Storage
    to BigQuery.

    Args:
      context: Airflow context that contains references to related objects to
        the task instance.
    """
    dataset_ref = self._bq_client.dataset(self._dataset_id)
    table_ref = dataset_ref.table(self._table_id)
    for blob in self.filter_blobs(self._exclude_prefix, self._dir_prefix):
      job = self._bq_client.load_table_from_uri(
          blob,
          table_ref,
          location=self._gcs_location,
          job_config=self._job_config)  # API request.
      job.result()  # Waits for table load to complete.
      if job.state == 'DONE':
        logger.info('Loaded "%d" rows to "%s".',
                    self._bq_client.get_table(table_ref).num_rows,
                    self._table_id)
