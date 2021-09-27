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

"""Setup file for the model package passed to ai platform.

  We expect the model package to be placed with tasks/transformer in the same
  parent directory. When calling ai-platform submit, use model.trainer as
  module-name and model as package-path.
"""

import setuptools

_REQUIRED_PACKAGES = [
    'gast==0.2.2',
    'tensorflow==2.4.1',
    'tensorflow-data-validation==0.13.0',
    'tensorflow-metadata==0.13.0',
    'tensorflow-transform==0.13.0',
]

_MODEL = 'model'
_TRANSFORMER = 'tasks/transformer'

if __name__ == '__main__':
  setuptools.setup(
      name=_MODEL,
      packages=[_MODEL, _TRANSFORMER],
      install_requires=_REQUIRED_PACKAGES)
