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
"""Feature columns configuration."""

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 10
# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 5

# All feature columns in dataset.
ALL_FEATURES = [
    'id_col', 'num_col1', 'num_col2', 'num_col3', 'num_col4', 'cat_col1',
    'cat_col2'
]

# Column with target values.
TARGET_FEATURE = 'num_col4'

# Column with unique ids.
ID_FEATURE = 'id_col'

# Columns to exclude from training.
EXCLUDED_FEATURES = [ID_FEATURE, 'num_col3']

# A feature to be exported along with prediction values.
FORWARD_FEATURE = ID_FEATURE

# Features with categorical values.
CATEGORICAL_FEATURES = ['cat_col1', 'cat_col2']

# Features with continuous numerical values.
NUMERIC_FEATURES = [
    col for col in ALL_FEATURES
    if col not in (EXCLUDED_FEATURES + CATEGORICAL_FEATURES + [TARGET_FEATURE])
]
