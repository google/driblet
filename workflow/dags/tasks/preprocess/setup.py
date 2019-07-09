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
"""Setup configuration for preprocess package."""

import setuptools

_NAME = 'preprocess'
_VERSION = '0.1'
_REQUIRED_PACKAGES = [
    'tensorflow-data-validation==0.11.0', 'tensorflow-metadata==0.9.0',
    'tensorflow-transform==0.11.0', 'protobuf==3.6.1'
]

if __name__ == '__main__':
  setuptools.setup(
      name=_NAME,
      version=_VERSION,
      packages=setuptools.find_packages(include=[_NAME]),
      install_requires=_REQUIRED_PACKAGES,
      exclude_package_data={'': [
          '*_test.py',
          '*/test_data/*',
      ]})
