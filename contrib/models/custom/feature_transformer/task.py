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
"""Feature transformer module."""
import argparse
import logging
import os
import shutil
import sys
import uuid

import apache_beam as beam
from google.cloud import bigquery
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from google.protobuf import text_format
from gps_building_blocks.cloud.utils import cloud_storage
from driblet.contrib.models.custom.feature_transformer import utils
from tensorflow_metadata.proto.v0 import schema_pb2

logging.basicConfig(level=logging.ERROR)


def parse_arguments():
  """Initialize command line parser using arparse.

  Returns:
    argparse.Namespace with arguments.
  """
  parser = argparse.ArgumentParser()

  # Flags related to Apache beam.
  parser.add_argument(
      '--runner',
      default='DirectRunner',
      type=str,
      help=('Where to run the pipeline - locally or on Cloud Dataflow. '
            'By default pipeline runs locally. Use `DataflowRunner` to run '
            'on the Cloud.'))

  # Flags related to transformation input datasets.
  parser.add_argument(
      '--features_config',
      default=None,
      type=str,
      required=True,
      help='Path to features configuration file (.cfg)')
  parser.add_argument(
      '--project_id',
      default=None,
      type=str,
      help='Google Cloud Platform project id.')
  parser.add_argument(
      '--data_source',
      default='csv',
      type=str,
      help='Data source: CSV file or BigQuery table (csv|bigquery).')
  parser.add_argument(
      '--skip_header_lines',
      default=1,
      type=int,
      help=(
          'Number of header lines to skip in CSV. By default it is 1 that '
          'skips column names, as they are given in features configuration file.'
      ))
  parser.add_argument(
      '--all_data',
      default=None,
      type=str,
      help=(
          'Path to CSV file (local or Cloud Storage) or BigQuery table '
          '(<cloud-project-id>.<dataset-name>.<table-name>)containing all data.'
      ))
  parser.add_argument(
      '--train_data',
      default=None,
      type=str,
      help='Path to CSV file (local or Cloud Storage) or BigQuery table '
      '(<cloud-project-id>.<dataset-name>.<table-name> containing training '
      'data.')
  parser.add_argument(
      '--eval_data',
      default=None,
      type=str,
      help='Path to CSV file (local or Cloud Storage) or BigQuery table '
      '(<cloud-project-id>.<dataset-name>.<table-name>) containing evaluation '
      'data.')
  parser.add_argument(
      '--predict_data',
      default=None,
      type=str,
      help='Path to CSV file (local or Cloud Storage) or BigQuery table '
      '(<cloud-project-id>.<dataset-name>.<table-name>) containing prediction '
      'data.')
  parser.add_argument(
      '--mode_train',
      default=False,
      type=bool,
      help=('If True, do transformation for all data (train, eval, predict). '
            'Otherwise, transform only predict data.'))
  parser.add_argument(
      '--schema_file',
      default='schema.pbtxt',
      type=str,
      help=('Schema file generated automatically by tensorflow_data_validation '
            'based on the whole dataset.'))

  # Flags related to transformation output.
  parser.add_argument(
      '--transform_dir',
      default=None,
      type=str,
      required=True,
      help='Local or Cloud Storage directory to store the transformer model.')
  parser.add_argument(
      '--output_dir',
      default=None,
      type=str,
      required=True,
      help='Local or Cloud Storage directory to store the transformed data.')
  parser.add_argument(
      '--region',
      default='asia-northeast1',
      type=str,
      help='Google Cloud region to run the Dataflow job in.')

  if os.environ['USER'] == 'jupyter':
    return parser.parse_args(sys.argv[1:])
  else:
    return parser.parse_args()


def _bq_preprocessing_fn(input_data, raw_feature_spec):
  """Callback function to preprocess raw input data from BigQuery table.

  Converts BOOLEAN values to STRING and assigns an empty list to NULL values.
  Other data types are returned as they are. All values in the dict are lists.
  This is required for the Tensorflow transformer to correctly transform and
  normalize the data, which is further consumed by the model (classification or
  regression).

  Args:
    input_data: A dictionary of raw input data fed by the Beam pipeline from a
      BigQuery table.
    raw_feature_spec: A dictionary of raw feature specs for input data generated
      based on schema proto.

  Returns:
    outputs: A dictionary of preprocessed data.
  """
  outputs = {}
  for key in raw_feature_spec:
    if input_data[key]:
      if isinstance(input_data[key], bool):
        outputs[key] = [str(input_data[key])]
      else:
        outputs[key] = [input_data[key]]
    else:
      outputs[key] = []
  return outputs


class _RecordBatchToPyDict(beam.PTransform):
  """Converts PCollections of PyArrow RecordBatch to python dicts."""

  def expand(self, pcoll):

    def format_values(instance):
      return {
          k: v.squeeze(0).tolist() if v is not None else None
          for k, v in instance.items()
      }

    return (pcoll | 'RecordBatchToDicts' >>
            beam.FlatMap(lambda x: x.to_pandas().to_dict(orient='records'))
            | 'FormatPyDictValues' >> beam.Map(format_values))


class _ReadData(beam.PTransform):
  """Wrapper class for reading CSV files (local or GCS) or BigQuery table."""

  def __init__(self, project_id, input_data, data_source, schema_file, mode,
               features_config, skip_header_lines):
    """Initializes _ReadData instance.

    Args:
      project_id: Google Cloud Platform project id.
      input_data: A path to a CSV file (local or Cloud Storage) or BigQuery
        table name specified as <cloud-project-id>.<dataset-name>.<table-name>.
      data_source: Type of data source: CSV file or BigQuery table. Expects
        either `csv` or `bigquery`.
      schema_file: Serialized Schema proto file.
      mode: One of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
      features_config: Features config.
      skip_header_lines: Number of header lines to skip.
    """
    self._project_id = project_id
    self._input_data = input_data
    self._data_source = data_source
    self._schema_file = schema_file
    self._mode = mode
    self._features_config = features_config
    self._skip_header_lines = skip_header_lines

  def expand(self, pvalue):
    """Reads data from local or Cloud Storage CSV or BigQuery Table.

    Args:
      pvalue: A Beam processing graph node.

    Returns:
      data: A PCollection that represents the data.
    """
    if self._data_source == 'csv':
      coder = utils.make_csv_coder(self._schema_file,
                                   self._features_config['all_features'],
                                   self._mode,
                                   self._features_config['target_feature'])
      raw_feature_spec = utils.get_raw_feature_spec(
          self._schema_file,
          mode=self._mode,
          target_feature=self._features_config['target_feature'])
      data = (
          pvalue.pipeline | 'ReadFromCSV' >> beam.io.ReadFromText(
              self._input_data,
              skip_header_lines=self._skip_header_lines,
              coder=beam.coders.BytesCoder())
          | 'RecordBatch' >> coder.BeamSource()
          | 'RecordBatchToPyDict' >> _RecordBatchToPyDict())
    else:
      query = 'SELECT * FROM `%s`;' % self._input_data
      raw_feature_spec = utils.get_raw_feature_spec(
          self._schema_file,
          mode=self._mode,
          target_feature=self._features_config['target_feature'])
      data = (
          pvalue.pipeline
          | 'ReadFromBigQuery' >> beam.io.gcp.bigquery.ReadFromBigQuery(
              query=query, project=self._project_id, use_standard_sql=True)
          | 'PreprocessBigQueryData' >> beam.Map(_bq_preprocessing_fn,
                                                 raw_feature_spec))
    return data


@beam.ptransform_fn
def _transform_and_write(pcollection, input_metadata, output_dir, transform_fn,
                         file_prefix):
  """Transforms data and writes results to local disk or Cloud Storage bucket.

  Args:
    pcollection: A PCollection represting the pipeline data.
    input_metadata: dataset_metadata.DatasetMetadata object of an input data.
    output_dir: Directory to write transformed dataset output.
    transform_fn: TensorFlow transform function.
    file_prefix: File prefix to add to output file.
  """
  shuffled_data = (pcollection | 'RandomizeData' >> beam.transforms.Reshuffle())
  (transformed_data,
   transformed_metadata) = (((shuffled_data, input_metadata), transform_fn)
                            | 'Transform' >> tft_beam.TransformDataset())
  coder = tft.coders.example_proto_coder.ExampleProtoCoder(
      transformed_metadata.schema)
  (transformed_data
   | 'SerializeExamples' >> beam.Map(coder.encode)
   | 'WriteExamples' >> beam.io.WriteToTFRecord(
       os.path.join(output_dir, file_prefix), file_name_suffix='.tfrecord'))


def _make_preprocessing_fn(features_config):
  """Creates a preprocessing function for tf.Transform module.

  Args:
    features_config: A dictionary of features configuration.

  Returns:
    preprocessing_fn: A preprocessing function.
  """

  def _preprocessing_fn(inputs):
    """Callback function for transforming inputs.

    Args:
      inputs: A dictionary of feature keys maped to `Tensor` or `SparseTensor`
        of raw features.

    Returns:
      outputs: Map from string feature keys to `Tensor` of transformed features.
    """
    outputs = {
        features_config['target_feature']:
            utils.preprocess_sparsetensor(
                inputs.pop(features_config['target_feature']))
    }
    # Push forward features without doing any preprocessing.
    if features_config['forward_features']:
      for key in features_config['forward_features']:
        outputs[key] = inputs.pop(key)
    # Normalize numeric feature columns with mean 0 and variance 1.
    for key in features_config['numeric_features']:
      outputs[utils.make_transformed_key(key)] = tft.scale_to_z_score(
          utils.preprocess_sparsetensor(inputs[key]))
    # Generate a vocabulary for categorical input feature and maps it to
    # an integer with this vocab.
    for key in features_config['categorical_features']:
      outputs[utils.make_transformed_key(
          key)] = tft.compute_and_apply_vocabulary(
              utils.preprocess_sparsetensor(inputs[key]),
              top_k=features_config['vocab_size'],
              num_oov_buckets=features_config['oov_size'])
    return outputs

  return _preprocessing_fn


def _transform_train_eval_features(project_id, pipeline, features_config,
                                   train_data, eval_data, data_source,
                                   transform_dir, output_dir, schema,
                                   skip_header_lines):
  """Analyzes and transforms train and evaluation dataset.

  Args:
    project_id: Google Cloud Platform project id.
    pipeline: An instance of Beam Pipeline.
    features_config: A dictionary of features configuration.
    train_data: Training input dataset.
    eval_data: Evaluation input dataset.
    data_source: Data source input type. Expects either `csv` or `bigquery`.
    transform_dir: A directory to write transformer model files.
    output_dir: A directory to write transformed dataset.
    schema: A text-serialized TensorFlow metadata schema for the input data.
    skip_header_lines: Number of header lines to skip.
  """

  train_raw_data = (
      pipeline | 'ReadTrainData' >> _ReadData(
          project_id, train_data, data_source, schema,
          tf.estimator.ModeKeys.TRAIN, features_config, skip_header_lines))
  eval_raw_data = (
      pipeline | 'ReadEvalData' >>
      _ReadData(project_id, eval_data, data_source, schema,
                tf.estimator.ModeKeys.EVAL, features_config, skip_header_lines))
  schema = utils.make_dataset_schema(schema, tf.estimator.ModeKeys.TRAIN,
                                     features_config['target_feature'])
  input_metadata = dataset_metadata.DatasetMetadata(schema)
  preprocessing_fn = _make_preprocessing_fn(features_config)
  logging.info('Creating new transformer model.')
  transform_fn = ((train_raw_data, input_metadata) |
                  ('Analyze' >> tft_beam.AnalyzeDataset(preprocessing_fn)))

  (transform_fn |
   ('WriteTransformFn' >> tft_beam.WriteTransformFn(transform_dir)))

  (train_raw_data | 'TransformAndWriteTraining' >> _transform_and_write(
      input_metadata, output_dir, transform_fn, 'train'))
  (eval_raw_data | 'TransformAndWriteEval' >> _transform_and_write(
      input_metadata, output_dir, transform_fn, 'eval'))


def _transform_predict_features(project_id, pipeline, features_config,
                                predict_data, data_source, output_dir, schema,
                                skip_header_lines):
  """Transforms prediction input dataset.

  Args:
    project_id: Google Cloud Platform project id.
    pipeline: An instance of Beam Pipeline.
    features_config: A dictionary of features configuration.
    predict_data: Prediction input dataset.
    data_source: Data source input type. Expects either `csv` or `bigquery`.
    output_dir: A directory to write transformed output.
    schema: A text-serialized TensorFlow metadata schema for the input data.
    skip_header_lines: Number of header lines to skip.
  """
  data_schema = utils.make_dataset_schema(schema, tf.estimator.ModeKeys.PREDICT,
                                          features_config['target_feature'])
  coder = tft.coders.example_proto_coder.ExampleProtoCoder(data_schema)

  raw_data = (
      pipeline | 'ReadPredictData' >> _ReadData(
          project_id, predict_data, data_source, schema,
          tf.estimator.ModeKeys.PREDICT, features_config, skip_header_lines))
  (raw_data
   | 'EncodePredictData' >> beam.Map(coder.encode)
   | 'WritePredictDataAsTFRecord' >> beam.io.WriteToTFRecord(
       os.path.join(output_dir, 'predict'), file_name_suffix='.tfrecord'))


def _get_bq_data(table_path: str) -> pd.DataFrame:
  """Reads BigQuery table into Pandas Dataframe.

  TODO(zmtbnv): Replace this method with tfx.StatisticsGen to be able to handle
  large datasets.

  Args:
    table_path: Full path to BigQuery table. Ex: 'project.dataset.table'.

  Returns:
    Pandas Dataframe.
  """
  client = bigquery.Client()
  sql = f'SELECT * FROM {table_path};'
  return client.query(sql).result().to_dataframe()


def _get_data_schema(data_source: str, data_path: str) -> schema_pb2.Schema:
  """Generates statistics for dataset and returns schema.

  Args:
    data_source: Type of data source: CSV file or BigQuery table. Expects either
      `csv` or `bigquery`.
    data_path: Full path to CSV or BigQuery table.

  Returns:
    A Schema protocol buffer for dataset.

  """
  if data_source == 'csv':
    stats = tfdv.generate_statistics_from_csv(data_location=data_path)
  else:
    df = _get_bq_data(table_path=data_path)
    stats = tfdv.generate_statistics_from_dataframe(dataframe=df)
  return tfdv.infer_schema(statistics=stats, infer_feature_shape=False)


def _get_pipeline_options(
    runner: str,
    project_id: str,
    region: str,
    output_dir: str,
    job_name: str = 'data-transform') -> beam.pipeline.PipelineOptions:
  """Constructs options for local and cloud pipeline.

  Args:
    runner: Environment to run beam workflow.
    project_id: Google Cloud Platform project id.
    region: Google Cloud region to run the Dataflow job in.
    output_dir: Local or Cloud Storage directory to store the transformed data.
    job_name: The name of the Cloud Dataflow job being executed.

  Returns:
    Instance of PipelineOptions with preset configuration.
  """
  options = {'temp_location': os.path.join(output_dir, 'tmp')}
  if runner == 'DataflowRunner':
    # Setup configuration script to run transformation on Cloud Dataflow.
    dataflow_setup_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'setup.py'))
    options['job_name'] = '{}-{}'.format(job_name, uuid.uuid1().hex)
    options['project'] = project_id
    options['region'] = region
    options['service_account_email'] = ''
    options['setup_file'] = dataflow_setup_file
  return beam.pipeline.PipelineOptions(flags=[], **options)


def main(args):
  # Directory to store pipeline's templorary files.
  tmp_dir = os.path.join(args.output_dir, 'tmp')

  schema = _get_data_schema(
      data_source=args.data_source, data_path=args.all_data)

  if args.runner == 'DataflowRunner':
    gcs_utils = cloud_storage.CloudStorageUtils(args.project_id)
    schema_path = os.path.join(args.transform_dir, args.schema_file)
    gcs_utils.write_to_path(text_format.MessageToString(schema), schema_path)
    pipeline_options = _get_pipeline_options(args.runner, args.project_id,
                                             args.region, args.output_dir)
  else:
    pipeline_options = _get_pipeline_options(args.runner, args.project_id,
                                             args.region, args.output_dir)
    if os.path.exists(args.transform_dir):
      logging.info('Removing existing transformed directory %s',
                   args.transform_dir)
      shutil.rmtree(args.transform_dir, ignore_errors=True)
    if not tf.io.gfile.exists(args.transform_dir):
      tf.io.gfile.makedirs(args.transform_dir)

    schema_file = os.path.join(args.transform_dir, args.schema_file)
    with tf.io.gfile.GFile(schema_file, 'w') as f:
      f.write(text_format.MessageToString(schema))
    logging.info('Generated schema: %s', schema_file)
    logging.info('Running pipeline on %s environment', args.runner)
  features_config = utils.parse_features_config(args.features_config)
  with beam.Pipeline(args.runner, options=pipeline_options) as pipeline:
    with tft_beam.Context(temp_dir=tmp_dir):
      # `mode=predict` should be used during the inference time.
      if args.mode_train:
        logging.info('Transforming training, evaluation and predict data.')
        _transform_predict_features(
            project_id=args.project_id,
            pipeline=pipeline,
            features_config=features_config,
            predict_data=args.predict_data,
            data_source=args.data_source,
            output_dir=args.output_dir,
            schema=schema,
            skip_header_lines=args.skip_header_lines)
        _transform_train_eval_features(
            project_id=args.project_id,
            pipeline=pipeline,
            features_config=features_config,
            train_data=args.train_data,
            eval_data=args.eval_data,
            data_source=args.data_source,
            transform_dir=args.transform_dir,
            output_dir=args.output_dir,
            schema=schema,
            skip_header_lines=args.skip_header_lines)
      else:
        logging.info('Transforming only prediction data.')
        _transform_predict_features(
            project_id=args.project_id,
            pipeline=pipeline,
            features_config=features_config,
            predict_data=args.predict_data,
            data_source=args.data_source,
            output_dir=args.output_dir,
            schema=schema,
            skip_header_lines=args.skip_header_lines)


if __name__ == '__main__':
  cli_args = parse_arguments()
  main(cli_args)
