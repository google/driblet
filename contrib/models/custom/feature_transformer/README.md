# Feature Transformer implementation guideline

This document provides an implementation guide for *data transformation* using
[TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started)
based on [Apache Beam](https://beam.apache.org/), which can run both locally or
on [Cloud Dataflow](https://cloud.google.com/dataflow/). Input data to
transformer can be either CSV file or BigQuery.

To run the pipeline, you need to do 2 steps:

1.  Setup the environment
2.  Run the pipeline

The following shows how to setup an environment based on input source: either
CSV or BigQuery table.

## Step 1: Configure environment variables

Before running the pipeline, you need to setup the environment. There are two
scenarios based on input data source:

### Case 1: CSV file

```bash
PROJECT=<gcp_project_id>
RUNNER=<runner>
JOB_NAME=<unique_job_name_for_cloud_dataflow>
DATA_SOURCE=csv
ALL_DATA=<path_to_all_data.csv>
TRAIN_DATA=<path_to_train_data.csv>
EVAL_DATA=<path_to_eval_data.csv>
PREDICT_DATA=<path_to_predict_data.csv>
TRANSFORM_DIR=<path_to_store_transformer_files>
OUTPUT_DIR=<path_to_store_transformed_files>
FEATURES_CONFIG=<path_to_features_config.cfg>
MODE_TRAIN=<true_or_false>
```

DESCRIPTION:

*   `PROJECT`: Google Cloud Platform project id that you can get from
    [Google Cloud Console](https://console.cloud.google.com).
*   `RUNNER`: Indicates where to run the data transformation pipeline. If you
    want to run on the Cloud Dataflow, set the value to `DataflowRunner`.
    Otherwise, set it to `DirectRunner` to run locally.
*   `JOB_NAME`: Job name for pipeline to run on Cloud DataFlow.
*   `DATA_SOURCE`: Indicates where input data comes from. Current version
    supports two options: `csv` or `bigquery`.
*   `FEATURES_CONFIG`: Features configuration `.cfg` file containing
    specifications for features.
*   `MODE_TRAIN`: [optional] If set true, pipeline will transform train and eval
    datasets. Otherwise, it transforms only prediction dataset.

Following four variables are input data sources. They can be local CSV files or
CSV files on Cloud Storage. If local CSV, indicate relative path:
`<path/to/file.csv>`. If CSV is stored in Cloud Storage, indicate full storage
bucket path: `gs://bucket/file.csv`

*   `ALL_DATA`: CSV containing all data.
    [Data schema](https://www.tensorflow.org/tfx/transform/get_started#data_formats_and_schema)
    is inferred automatically based on this and saved as proto buffer file,
    which is used during transformation process and inference time.
*   `TRAIN_DATA`: CSV containing only training data.
*   `EVAL_DATA`: CSV containing only eval data.
*   `PREDICT_DATA`: CSV containing only predict data.
*   `TRANSFORM_DIR`: Path to local directory or Cloud Storage path to store
    [TensorFlow transform model](https://www.tensorflow.org/tfx/transform/get_started)
    and related files created during transformation process.
*   `OUTPUT_DIR`: Directory path transformed data will be saved. It can be local
    directory or Cloud Storage folder.

EXAMPLE:

```bash
PROJECT=my-cloud-project
RUNNER=DirectRunner
JOB_NAME=transform-job
DATA_SOURCE=csv
ALL_DATA=test_data/data_all.csv
TRAIN_DATA=test_data/data_train.csv
EVAL_DATA=test_data/data_eval.csv
PREDICT_DATA=test_data/data_predict.csv
TRANSFORM_DIR=~/sample_output/transformer
OUTPUT_DIR=~/sample_output/dataset
FEATURES_CONFIG=features_config.cfg
MODE_TRAIN=True
```

### Case 2: BigQuery table

All environment variables are similar to above setup, except for the following:

*   `DATA_SOURCE` needs to be set to `bigquery`.
*   All data path variables must indicate the full path to the BigQuery table.

```bash
DATA_SOURCE=bigquery
ALL_DATA=$PROJECT.<dataset_name>.<table_name_for_all data>
TRAIN_DATA=$PROJECT.<dataset_name>.<table_name_for_training>
EVAL_DATA=$PROJECT.<dataset_name>.<table_name_for_evaluation>
PREDICT_DATA=$PROJECT.<dataset_name>.<table_name_for_prediction>
```

## Step 2: Run the pipeline

There are two options you can run the pipeline: locally or on the Cloud.

### Option 1: Run locally

In order to run locally, set the RUNNER variable to DirectRunner:
`RUNNER=DirectRunner`.

NOTE: All input data and output directories can be *either local or Cloud
Storage bucket*.

```bash
python feature_transformer/feature_transformer.py -- \
  --project $PROJECT \
  --runner $RUNNER \
  --job_name $JOB_NAME \
  --all_data $ALL_DATA \
  --train_data $TRAIN_DATA \
  --eval_data $EVAL_DATA \
  --predict_data $PREDICT_DATA \
  --transform_dir $TRANSFORM_DIR \
  --output_dir $OUTPUT_DIR \
  --data_source $DATA_SOURCE \
  --features_config $FEATURES_CONFIG \
  --mode_train $MODE_TRAIN
```

### Option 2: Run on Cloud DataFlow

In order to run on the Cloud, set RUNNER to DataflowRunner:
`RUNNER=DataflowRunner`.

NOTE: All input data and output directories *should be on Cloud Storage bucket*.

```bash
python -m workflow.dags.tasks.preprocess.transformer \
  --project $PROJECT \
  --runner $RUNNER \
  --job_name $JOB_NAME \
  --all_data $ALL_DATA \
  --train_data $TRAIN_DATA \
  --eval_data $EVAL_DATA \
  --predict_data $PREDICT_DATA \
  --transform_dir $TRANSFORM_DIR \
  --output_dir $OUTPUT_DIR \
  --data_source $DATA_SOURCE \
  --features_config $FEATURES_CONFIG \
  --mode_train $MODE_TRAIN
```

## Other options

The above runs will transform whole dataset: train, eval, predict. But during
inference time, you may only need a transformation only for prediction data. To
do so, supply an additional `--mode=predict` flag. If set, only the
`PREDICT_DATA` environment variable needs to have path to data. Others
(`TRAIN_DATA`, `EVAL_DATA`) can be set to None.
