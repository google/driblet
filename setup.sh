#!/bin/bash
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

# Driblet setup script.

set -e

# Set Python virtual environment path.
VIRTUALENV_PATH=$HOME/"driblet-venv"
PYTHONPATH=src/plugins:$PYTHONPATH
export PYTHONPATH

# Create virtual environment with python3.
if [[ ! -d "${VIRTUALENV_PATH}" ]]; then
  virtualenv -p python3 "${VIRTUALENV_PATH}"
fi

# Activate virtual environment.
source ~/driblet-venv/bin/activate
# Install Python dependencies.
pip install -r requirements.txt


read -p "Provide service account name: " SERVICE_ACCOUNT_NAME
echo "Select modeling option (Ex: 1, 2 or 3): "
modeling_platforms=("BigQueryMl" "AutoMl" "TensorFlow")
select modeling_platform in "${modeling_platforms[@]}"
do
  case $modeling_platform in
    "AutoMl") break;;
    "BigQueryMl")  break;;
    "TensorFlow") break;;
    *) echo "Invalid option $REPLY";;
  esac
done
echo "$modeling_platform is selected."

# Setup cloud environment.
python cloud_env_setup.py \
  --project_id="$GOOGLE_CLOUD_PROJECT" \
  --service_account_name="$SERVICE_ACCOUNT_NAME" \
  --modeling_platform="$modeling_platform"
