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

"""Tests for driblet.plugins.pipeline_plugins.operators.dataflow_py3_operator."""

import os
import unittest
from airflow.contrib.operators import dataflow_operator
import mock
from parameterized import parameterized

from driblet.contrib.plugins.hooks import dataflow_py3_hook
from driblet.contrib.plugins.operators import dataflow_py3_operator


class TestDataFlowOperator(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self._mock_dataflow_hook = mock.create_autospec(
        dataflow_py3_hook.DataFlowHook)
    dataflow_operator.DataFlowPythonOperator.__init__ = mock.Mock()
    dataflow_operator.GoogleCloudBucketHelper = mock.Mock()

  @parameterized.expand([
      ('python_path', 'test_python_path', 'python_path:test_python_path'),
      ('', 'test_python_path', 'test_python_path'),
      ('test_python_path', 'test_python_path', 'test_python_path'),
      ('test:foo:bar', 'test', 'test:foo:bar'),
  ])
  def test_dataflow_operator(self, sys_pypath, beam_pypath, combined_pypath):
    with mock.patch('os.environ', {'PYTHONPATH': sys_pypath}):
      op = dataflow_py3_operator.DataFlowPythonOperator(
          python_path=beam_pypath,
          dataflow_hook=self._mock_dataflow_hook,
      )

      self.assertEqual(os.environ['PYTHONPATH'], combined_pypath)
      self.assertIsNotNone(op)

      op.task_id = 'test-task'
      op.py_file = 'df_script.py'
      op.gcp_conn_id = 'fake_conn_id'
      op.delegate_to = None
      op.dataflow_default_options = {}
      op.options = {}
      op.py_options = {}

      op.execute({})
      self._mock_dataflow_hook.start_python_dataflow.assert_called()


if __name__ == '__main__':
  unittest.main()
