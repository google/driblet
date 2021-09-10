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
-- This is an example script for predicting based on BQML model.
--
-- Query expects following parameters:
-- `features_input_table`: Full table path containing features data. Ex: project.dataset.features.
-- `prediction_output_table`: Full table path to store predicton results. Ex: project.dataset.output.
-- `model_name`: Full table path to BQML model. Ex: project.dataset.model.

CREATE OR REPLACE TABLE `{prediction_output_table}`
AS (
  SELECT
    *
  FROM
    ML.PREDICT(
      MODEL `{model_name}`,
      (
        SELECT
          col1, col2, col3
        FROM
          `{features_input_table}`
      ))
);
