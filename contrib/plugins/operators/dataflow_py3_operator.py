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

"""A Dataflow Operator to run py3 jobs.

TODO(codescv): b/143115359
Since the official dataflow operator does not support python 3 yet(hard-coded
python 2), we patch it to support python 3. Once the official implementation
supports python 3 this module will be removed.
"""

import os
import re

from airflow.contrib.operators import dataflow_operator

from driblet.contrib.plugins.hooks import dataflow_py3_hook


def _camel_to_snake(name: str) -> str:
  """Convert argument name from lowerCamelCase to snake case.

  Args:
    name: Argument name.

  Returns:
    Converted name.
  """
  return re.sub(r'[A-Z]', lambda x: '_' + x.group(0).lower(), name)


class DataFlowPythonOperator(dataflow_operator.DataFlowPythonOperator):
  """A Dataflow Operator to run py3 jobs.

  This operator patches `dataflow_operator.DataFlowPythonOperator` and call
  `dataflow_py3_hook.DataFlowHook` to start DataFlow jobs in python 3.
  """

  def __init__(self,
               python_path='',
               dataflow_hook=None,
               **kwargs):
    """Constructor.

    Args:
      python_path: Set PYTHONPATH to include this path before calling DataFlow.
      dataflow_hook: The DataFlow hook to use. If none, will automatically
        create.
      **kwargs: Additional args for super class.
    """
    old_pypath = os.environ.get('PYTHONPATH', '')
    if not old_pypath:
      pypath = python_path
    elif python_path in old_pypath:
      pypath = old_pypath
    else:
      pypath = old_pypath + ':' + python_path
    os.environ['PYTHONPATH'] = pypath
    self.dataflow_hook = dataflow_hook
    super().__init__(**kwargs)

  def execute(self, context):
    """Execute the python dataflow job."""
    bucket_helper = dataflow_operator.GoogleCloudBucketHelper(
        self.gcp_conn_id, self.delegate_to)
    self.py_file = bucket_helper.google_cloud_to_local(self.py_file)
    if self.dataflow_hook is None:
      self.dataflow_hook = dataflow_py3_hook.DataFlowHook(
          gcp_conn_id=self.gcp_conn_id,
          delegate_to=self.delegate_to,
          poll_sleep=self.poll_sleep)
    dataflow_options = self.dataflow_default_options.copy()
    dataflow_options.update(self.options)

    formatted_options = {_camel_to_snake(key): dataflow_options[key]
                         for key in dataflow_options}
    self.dataflow_hook.start_python_dataflow(
        self.task_id, formatted_options,
        self.py_file, self.py_options)
