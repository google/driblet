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
"""Setup module for local development.

It needs to be run before data preprocessing and training.
Run from project's root directory with `develop` suffix.
Example: `python setup.py develop`
"""

import os
import setuptools

_NAME = 'driblet'
_VERSION = '0.1.0'
_REQUIREMENTS_DOC = 'requirements.txt'


def main():
  # Set env variable for Airflow installation to avoid the GPL version
  os.environ['SLUGIFY_USES_TEXT_UNIDECODE'] = 'yes'

  with open(_REQUIREMENTS_DOC) as f:
    py_dependencies = f.read().splitlines()
    setuptools.setup(
        name=_NAME,
        version=_VERSION,
        packages=setuptools.find_packages(include=[_NAME]),
        install_requires=py_dependencies,
        exclude_package_data={'': [
            '*_test.py',
            '*/test_data/*',
        ]})


if __name__ == '__main__':
  main()
