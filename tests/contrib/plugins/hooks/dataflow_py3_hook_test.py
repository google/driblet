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

"""Tests for driblet.plugins.pipeline_plugins.hooks.dataflow_py3_hook."""

import unittest

from airflow.contrib.hooks import gcp_dataflow_hook
import mock

from driblet.contrib.plugins.hooks import dataflow_py3_hook


class TestDataFlowHook(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    gcp_dataflow_hook.DataFlowHook.__init__ = mock.Mock(return_value=None)
    self._mock_start_dataflow = mock.Mock(return_value=None)
    gcp_dataflow_hook.DataFlowHook._start_dataflow = self._mock_start_dataflow

  def test_dataflow_hook(self):
    hook = dataflow_py3_hook.DataFlowHook()
    task_id = 'test-task-id'
    hook.start_python_dataflow(task_id, {}, None, [])

    self.assertIsNotNone(hook)
    self._mock_start_dataflow.assert_called()


if __name__ == '__main__':
  unittest.main()
