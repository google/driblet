-- Copyright 2021 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
--
-- This is example sql script for generating custom features.
--
-- Query expects following parameters:
-- `raw_input_table`: Full table path containing raw data. Ex: project.dataset.raw.
-- `features_output_table`: Full table path to store features. Ex: project.dataset.features.

CREATE OR REPLACE TABLE `{features_output_table}`
AS (
  SELECT
    feature1,
    feature2
  FROM
    `{raw_input_table}`
);
