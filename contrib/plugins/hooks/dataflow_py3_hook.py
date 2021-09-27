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

"""Hook for running py3 Dataflow jobs.

TODO(codescv): b/143115359
Temporarily patching DataFlowHook to call python 3, once the official one
supports python 3 this will be removed.
"""

from typing import Any, Mapping, List, Dict
from airflow.contrib.hooks import gcp_dataflow_hook


def _label_formatter(labels_dict: Mapping[str, str]) -> List[str]:
  """Format labels dict to command line args.

  Args:
    labels_dict: Labels dictionary.
  Returns:
    Formatted label arguments.
  """
  return ['--labels={}={}'.format(key, value)
          for key, value in labels_dict.items()]


class DataFlowHook(gcp_dataflow_hook.DataFlowHook):
  """Hook for running py3 Dataflow jobs.

  Patch `gcp_dataflow_hook.DataFlowHook` to use python 3.
  """

  def start_python_dataflow(self,
                            task_id: str,
                            variables: Dict[str, str],
                            dataflow: Any,
                            py_options: List[str],
                            append_job_name: bool = True) -> None:
    """Start python DataFlow job.

    Args:
      task_id: The task id for the DataFlow job.
      variables: DataFlow options(key->value).
      dataflow: The DataFlow python file to be run.
      py_options: Additional python arguments.
      append_job_name: Whether to append unique random id to job name.
    """
    name = self._build_dataflow_job_name(task_id, append_job_name)
    variables['job_name'] = name

    self._start_dataflow(task_id, variables, name,
                         ['python3'] + py_options + [dataflow],
                         _label_formatter)
