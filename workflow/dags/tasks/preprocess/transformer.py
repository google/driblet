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
"""Input data transformer module."""

import argparse
import logging
import os
import shutil
import sys

import apache_beam as beam
import data_pipeline_utils as utils
import features_config
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_data_validation as tfdv
import tensorflow_transform as transform
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from google.protobuf import text_format

# Default Logger to log to the console.
logger = logging.getLogger(__name__)

_TRAIN_PREFIX = 'train'
_EVAL_PREFIX = 'eval'
_PREDICT_PREFIX = 'predict'
_FILE_NAME_SUFFIX = '.tfrecord'
_SCHEMA_FILE = 'schema.pbtxt'


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: A list of command line arguments.

  Returns:
    The parsed arguments returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      description='Runs preprocessing for input raw data.')
  parser.add_argument(
      '--runner',
      help='Runner option where to run the transformation.',
      choices=['DirectRunner', 'DataflowRunner'],
      default='DirectRunner')
  parser.add_argument('--project', help='GCP project id.')
  parser.add_argument(
      '--all-data',
      help='Path to CSV file (local or Cloud Storage) containing all data.')
  parser.add_argument(
      '--train-data',
      help='Path to CSV file (local or Cloud Storage) or BigQuery table'
      ' containing training data.')
  parser.add_argument(
      '--eval-data',
      help='Path to CSV file (local or Cloud Storage) or BigQuery table '
      ' containing evaluation data.')
  parser.add_argument(
      '--predict-data',
      help='Path to CSV file (local or Cloud Storage) or BigQuery table '
      ' containing prediction data.')
  parser.add_argument(
      '--data-source',
      help='Type of data source: CSV file or BigQuery table (csv|bigquery).',
      choices=['csv', 'bigquery'],
      default='bigquery',
      required=True)
  parser.add_argument(
      '--mode',
      choices=['train', 'predict'],
      default='train',
      help='If train, do transformation for all data (train, eval, predict). '
      'Otherwise, transform only predict data.')
  parser.add_argument(
      '--transform-dir', help='Directory to store transformer model.')
  parser.add_argument(
      '--output-dir', help='Directory to store transformed data.')
  parser.add_argument(
      '--job_name',
      help='Dataflow Runner job name dynamically created by Airflow.')
  args, _ = parser.parse_known_args(args=argv[1:])
  return args


def preprocessing_fn(inputs):
  """Callback function for transforming inputs.

  Args:
    inputs: A dict of feature keys maped to `Tensor` or `SparseTensor` of raw
      features.

  Returns:
    Map from string feature keys to `Tensor` of transformed features.
  """
  outputs = {
      features_config.TARGET_FEATURE:
          utils.preprocess_sparsetensor(
              inputs.pop(features_config.TARGET_FEATURE))
  }
  outputs[features_config.ID_FEATURE] = inputs.pop(features_config.ID_FEATURE)
  for key in features_config.NUMERIC_FEATURES:
    outputs[utils.make_transformed_key(key)] = transform.scale_to_z_score(
        utils.preprocess_sparsetensor(inputs[key]))
  for key in features_config.CATEGORICAL_FEATURES:
    outputs[utils.make_transformed_key(
        key)] = transform.compute_and_apply_vocabulary(
            utils.preprocess_sparsetensor(inputs[key]),
            top_k=features_config.VOCAB_SIZE,
            num_oov_buckets=features_config.OOV_SIZE)
  return outputs


def bq_preprocessing_fn(input_data, raw_feature_spec):
  """Callback function to preprocess raw input data from BigQuery table.

  It converts BOOLEAN values to STRING and assigns empty list to NULL values.
  Other data types are returned as they are. This is required for Tensorflow
  transformer to correctly transform and normalize the data, which is further
  consumed by classification model.

  Args:
    input_data: A dictionary of raw input data fed by Beam pipeline from
      BigQuery table.
    raw_feature_spec: A dictionary of raw feature spec for input data generated
      based on schema proto.

  Returns:
    outputs: A dictionary of preprocessed data.
  """
  outputs = {}
  for key in raw_feature_spec:
    if isinstance(input_data[key], bool):
      outputs[key] = [str(input_data[key])]
    elif not input_data[key]:
      outputs[key] = []
    else:
      outputs[key] = [input_data[key]]
  return outputs


class ReadData(beam.PTransform):
  """Wrapper for reading CSV files (local or GCS) or BigQuery table."""

  def __init__(self, input_data, data_source, schema_file, mode):
    """Initializes ReadData instance.

    Args:
      input_data: A path to a CSV file (local or Cloud Storage) or BigQuery
        table name specified as <cloud-project-id>.<dataset-name>.<table-name>.
      data_source: Type of data source: CSV file or BigQuery table. Expects
        either `csv` or `bigquery`.
      schema_file: Serialized Schema proto file.
      mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
    """
    self._input_data = input_data
    self._data_source = data_source
    self._schema_file = schema_file
    self._mode = mode

  def expand(self, pvalue):
    """Reads data from local/GCS CSV or BigQuery Table.

    Args:
      pvalue: A Beam processing graph node.

    Returns:
      data: A PCollection that represents the data.
    """
    if self._data_source == 'csv':
      coder = utils.make_csv_coder(self._schema_file, mode=self._mode)
      data = (
          pvalue.pipeline
          | 'ReadFromCSV' >> beam.io.ReadFromText(
              self._input_data, skip_header_lines=1)
          | 'ParseCSV' >> beam.Map(coder.decode))
    else:
      query = 'SELECT * FROM `%s`;' % self._input_data
      raw_feature_spec = utils.get_raw_feature_spec(
          self._schema_file, mode=self._mode)
      data = (
          pvalue.pipeline
          | 'ReadFromBigQuery' >> beam.io.Read(
              beam.io.BigQuerySource(query=query, use_standard_sql=True))
          | 'PreprocessBigQueryData' >> beam.Map(bq_preprocessing_fn,
                                                 raw_feature_spec))
    return data


@beam.ptransform_fn
def transform_and_write(pcollection, input_metadata, output_dir, transform_fn,
                        file_prefix):
  """Transforms data and writes results to local disc or Cloud Storage bucket.

  Args:
    pcollection: Pipeline data.
    input_metadata: DatasetMetadata object for given input data.
    output_dir: Directory to write transformed output.
    transform_fn: TensorFlow transform function.
    file_prefix: File prefix to add to output file.
  """
  shuffled_data = (pcollection | 'RandomizeData' >> beam.transforms.Reshuffle())
  (transformed_data,
   transformed_metadata) = (((shuffled_data, input_metadata), transform_fn)
                            | 'Transform' >> tft_beam.TransformDataset())
  coder = example_proto_coder.ExampleProtoCoder(transformed_metadata.schema)
  (transformed_data
   | 'SerializeExamples' >> beam.Map(coder.encode)
   | 'WriteExamples' >> beam.io.WriteToTFRecord(
       os.path.join(output_dir, file_prefix),
       file_name_suffix=_FILE_NAME_SUFFIX))


def transform_train_and_eval(pipeline, train_data, eval_data, data_source,
                             transform_dir, output_dir, schema):
  """Analyzes and transforms data.

  Args:
    pipeline: Beam Pipeline instance.
    train_data: Training CSV data.
    eval_data: Evaluation CSV data.
    data_source: Input data source - path to CSV file or BigQuery table. Expects
      either `csv` or `bigquery`.
    transform_dir: Directory to write transformed output. If this directory
      exists beam loads `transform_fn` instead of computing it again.
    output_dir: Directory to write transformed output.
    schema: A text-serialized TensorFlow metadata schema for the input data.
  """
  train_raw_data = (
      pipeline | 'ReadTrainData' >> ReadData(train_data, data_source, schema,
                                             tf.estimator.ModeKeys.TRAIN))
  eval_raw_data = (
      pipeline | 'ReadEvalData' >> ReadData(eval_data, data_source, schema,
                                            tf.estimator.ModeKeys.EVAL))
  schema = utils.make_dataset_schema(schema, mode=tf.estimator.ModeKeys.TRAIN)
  input_metadata = dataset_metadata.DatasetMetadata(schema)
  logger.info('Creating new transform model.')
  transform_fn = ((train_raw_data, input_metadata)
                  | ('Analyze' >> tft_beam.AnalyzeDataset(preprocessing_fn)))

  (transform_fn
   | ('WriteTransformFn' >> tft_beam.WriteTransformFn(transform_dir)))

  (train_raw_data
   | 'TransformAndWriteTraining' >> transform_and_write(
       input_metadata, output_dir, transform_fn, _TRAIN_PREFIX))
  (eval_raw_data
   | 'TransformAndWriteEval' >> transform_and_write(input_metadata, output_dir,
                                                    transform_fn, _EVAL_PREFIX))


def transform_predict(pipeline, predict_data, data_source, output_dir, schema):
  """Transforms prediction input data.

  Args:
    pipeline: Beam Pipeline instance.
    predict_data: Prediction csv data.
    data_source: Input data source - path to CSV file or BigQuery table. Expects
      either `csv` or `bigquery`.
    output_dir: Directory to write transformed output.
    schema: A text-serialized TensorFlow metadata schema for the input data.
  """
  data_schema = utils.make_dataset_schema(
      schema, mode=tf.estimator.ModeKeys.PREDICT)
  coder = example_proto_coder.ExampleProtoCoder(data_schema)

  raw_data = (
      pipeline
      | 'ReadPredictData' >> ReadData(predict_data, data_source, schema,
                                      tf.estimator.ModeKeys.PREDICT))
  (raw_data
   | 'EncodePredictData' >> beam.Map(coder.encode)
   | 'WritePredictDataAsTFRecord' >> beam.io.WriteToTFRecord(
       os.path.join(output_dir, _PREDICT_PREFIX), file_name_suffix='.tfrecord'))


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  schema_file = os.path.join(args.transform_dir, _SCHEMA_FILE)
  if args.runner == 'DataflowRunner':
    schema = utils.read_schema(schema_file)
    dataflow_options = {
        'job_name':
            args.job_name,
        'project':
            args.project,
        'service_account_email':
            '',
        'setup_file':
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), 'setup.py')),
        'temp_location':
            os.path.join(args.output_dir, 'tmp')
    }
    pipeline_options = beam.pipeline.PipelineOptions(
        flags=[], **dataflow_options)
  else:
    pipeline_options = beam.pipeline.PipelineOptions(None)

    if os.path.exists(args.transform_dir):
      logger.info('Removing existing directory %s', args.transform_dir)
      shutil.rmtree(args.transform_dir)

    stats = tfdv.generate_statistics_from_csv(data_location=args.all_data)
    schema = tfdv.infer_schema(statistics=stats, infer_feature_shape=False)
    if not file_io.file_exists(args.transform_dir):
      file_io.recursive_create_dir(args.transform_dir)
    with file_io.FileIO(schema_file, 'w') as f:
      f.write(text_format.MessageToString(schema))
    logger.info('Generated %s', schema_file)
    logger.info('Running pipeline on %s environment', args.runner)

  with beam.Pipeline(args.runner, options=pipeline_options) as pipeline:
    with tft_beam.Context(temp_dir=os.path.join(args.output_dir, 'tmp')):
      # `mode=predict` should be used during inference time
      if args.mode == 'predict':
        logger.info('Transforming only prediction data.')
        transform_predict(
            pipeline=pipeline,
            predict_data=args.predict_data,
            data_source=args.data_source,
            output_dir=args.output_dir,
            schema=schema)
      else:
        logger.info('Transforming both training, evaluation and predict data.')
        transform_predict(
            pipeline=pipeline,
            predict_data=args.predict_data,
            data_source=args.data_source,
            output_dir=args.output_dir,
            schema=schema)
        transform_train_and_eval(
            pipeline=pipeline,
            train_data=args.train_data,
            eval_data=args.eval_data,
            data_source=args.data_source,
            transform_dir=args.transform_dir,
            output_dir=args.output_dir,
            schema=schema)


if __name__ == '__main__':
  main()
