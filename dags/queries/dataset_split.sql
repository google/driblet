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

-- This is a template script used to split BigQuery dataset into train, eval and test datasets used
-- for Tensorflow modeling.
--
-- Query expects following parameters:
-- `bq_input_table`: Full path to BigQuery table. Ex: project.dataset.table.
-- `id_column`: ID column that will be used to filter rows. Ex: 'user_id', 'gclid' etc.


-- Returns the remainder of the division of X by Y.
--
-- @param id_column: Id column name to use as a key for calculating the remainder.
-- @proportion_lower_bound: Lower bound value for dataset proportion. Ex: 0, 10, 50 etc.
-- @proportion_upper_bound: Upper bound value for dataset proportion. Ex: 0, 10, 100 etc.
CREATE TEMP FUNCTION GetRemainder(id_column STRING)
RETURNS INT64
AS (
  MOD(ABS(FARM_FINGERPRINT(CAST(id_column AS string))), 100)
);


SELECT
  *
FROM
  `{bq_input_table}`
WHERE
  GetRemainder({id_column}) >= {proportion_lower_bound}
  AND GetRemainder({id_column}) < {proportion_upper_bound};
