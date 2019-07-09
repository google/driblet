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
#!/bin/bash

set -e

VIRTUALENV_PATH=$HOME/"driblet-venv"

# TODO(azamat): migrate to python3.7 as Apache Beam fully supports Python 3
# Refer to current status: https://jira.apache.org/jira/browse/BEAM-1251?subTaskView=unresolved.
# Create virtual environment with python2.7
if [[ ! -d "${VIRTUALENV_PATH}" ]]; then
  virtualenv -p python2.7 "${VIRTUALENV_PATH}"
fi
