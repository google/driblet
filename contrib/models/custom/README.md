# TensorFlow Estimator based canned model templates

**This is not an official Google product.**

## Overview

This module provides high level API for four templates that can be used to train
`Regression` or `Classification` types of models:

1.  **Regressors:**

    1.  DNNRegressor -
        [tf.estimator.DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor)
        based regressor template.
    2.  DNNLinearCombinedRegressor -
        [tf.estimator.DNNLinearCombinedRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedRegressor)
        based regressor template.

2.  **Classifiers:**

    1.  Classifier -
        [tf.estimator.DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)
        based classifier template.
    2.  CombinedClassifier -
        [tf.estimator.DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)
        based classifier template.

## Package structure

The package consists of three key modules (excluding tests):

```
canned_models
  ├── input_functions.py
  ├── models.py
  ├── trainer.py
```

*   `input_functions.py`: Does two things:
    1.  Creates
        [feature columns](https://www.tensorflow.org/guide/feature_columns) for
        the model.
    2.  Parses TFRecord files and generates batches of input data as tensors
        according to the feature columns definition.
*   `models.py`: Creates one of the
    [Estimators](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
    specified in [*Overview*](#overview) section above.
*   `trainer.py`: It is a main module to trigger training and evaluation.

The only module you need to use is `trainer.py` which handles all tasks required
to train your model:

1.  Read and create batches for input datasets.
2.  Train and evaluate the model.
3.  Save the model that can be deployed on AI Platform.

## Step by step implementation guide

### Step 1: Data preprocessing

#### 1. Configure features

Edit
[features_config.cfg](/third_party/professional_services/solutions/driblet/contrib/models/features_config.cfg)
to configure feature columns in your dataset. This file contains feature names
for dummy dataset based on `test_data/all_data.csv`.

Modify following fields in `features_config.cfg` file to to match your dataset
features:

*   `target_feature`: Column with target values
*   `categorical_features`: Features with categorical values.
*   `numeric_features`: Features with numerical values.
*   `forward_features`: Features to be exported along with prediction values.
*   `vocab_size`: Vocabulary size for categorical features.
*   `oov_size`: Out Of Vocabulary buckets in which unrecognized vocab features
    are hashed.

#### 2. Run preprocessing pipeline

Dataset needs to be preprocessed to be able to train the model. Transform your
dataset (`CSV` or `BigQuery table`) to `.tfrecord` format. This can be done
easily by following steps described in
[Feature Transformer implementation guide](/third_party/professional_services/solutions/driblet/feature_transformer/README.md).

NOTE: Data preprocessing pipeline expects the dataset already be split into
`train, eval and test` datasets. If your data is in BigQuery, you can use steps
described in
[this page](https://www.oreilly.com/learning/repeatable-sampling-of-data-sets-in-bigquery-for-machine-learning).
Otherwise you can use
[Tensorflow Datasets Splits API](https://www.tensorflow.org/datasets/splits).

### Step 3: Train and deploy the model

Model training and deploying involves 3 steps:

1.  [Environment setup](#31-environment-setup)
2.  [Train the model](#32-train-the-model)
3.  [Deploy the model](#33-deploy-the-model)

#### 3.1: Environment setup

Before starting training following env variables need to be set.

Input files:

*   `features_config_file`: Path to features configuration file (.cfg).
*   `train_data`: GCS or local paths to training data.
*   `eval_data`: GCS or local paths to evaluation data.
*   `schema_file`: File holding the schema for the input data.

Required directories:

*   `transform_dir`: Tf-transform directory with model from preprocessing step.
*   `job_dir`: GCS location to write checkpoints and export models.

Model name and type:

*   `model_name`: Name of the model to save.
*   `estimator_type`: Type of the estimator. Should be one of [`Regressor`,
    `CombinedRegressor`, `Classifier`, `CombinedClassifier`].

Model hyperparameters:

*   `train_steps`: Count of steps to run the training job for.
*   `train_batch_size`: Train batch size.
*   `eval_steps`: Number of steps to run evaluation for at each checkpoint.
*   `eval_batch_size`: Eval batch size.
*   `num_epochs`: Number of epochs.
*   `first_layer_size`: Size of the first layer.
*   `num_layers`: Number of NN layers.
*   `save_checkpoints_steps`: Save checkpoints every N steps.
*   `keep_checkpoint_max`: Maximum number of recent checkpoint files to keep.
*   `exports_to_keep`: Number of model exports to keep.
*   `start_delay_secs`: Start evaluating after N seconds.
*   `throttle_secs`: Evaluate every N seconds.
*   `dnn_optimizer`: Optimizer for DNN model.
*   `dnn_dropout`: Dropout value (float) for DNN.
*   `linear_optimizer`: Optimizer for linear model.

Prediction output customization:

*   `include_prediction_class`: If set True, classification prediction output
    will include predicted classes. Otherwise, model outputs only probability.
    It's `False` by default.
*   `probability_output_key`: Key name for output probability value. Its
    `probability` by default.
*   `prediction_output_key`: Key name for output prediction value. Its
    `prediction` by default.

TODO(zmtbnv): Test multiple input tfrecord files and update example.

EXAMPLE:

```bash
export MODEL_DIR=~/sample_output/model
export ESTIMATOR_TYPE=Regressor
export TRAIN_DATA=~/sample_output/dataset/train-00000-of-00001.tfrecord
export EVAL_DATA=~/sample_output/dataset/eval-00000-of-00001.tfrecord
export TRANSFORM_DIR=~/sample_output/transformer
export SCHEMA=~/sample_output/transformer/schema.pbtxt
export TRAIN_STEPS=100
export TRAIN_BATCH_SIZE=10
export EVAL_STEPS=100
export EVAL_BATCH_SIZE=10
export NUM_EPOCHS=1
export SAVE_CHECKPOINTS_STEP=2
export KEEP_CHECKPOINT_MAX=1
export FIRST_LAYER_SIZE=4
export NUM_LAYERS=2
export DNN_OPTIMIZER=Adam
export LINEAR_OPTIMIZER=Ftrl
export INCLUDE_PREDICTION_CLASS=True
export PROBABILITY_OUTPUT_KEY=probability
export PREDICTION_OUTPUT_KEY=prediction
```

#### 3.2: Train the model

Model can be trained locally or on the
[Cloud AI Platform](https://cloud.google.com/ml-engine/docs/tensorflow/training-overview).

##### 3.2.1: Train locally

From root directory of the project, run following:

*NOTE: hyperparameters are set arbitrarily.*

```bash
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --estimator_type $ESTIMATOR_TYPE \
    --train-data $TRAIN_DATA \
    --eval-data $EVAL_DATA \
    --transform-dir $TRANSFORM_DIR \
    --schema-file $SCHEMA \
    --train-steps $TRAIN_STEPS \
    --eval-steps $TRAIN_BATCH_SIZE \
    --eval-batch-size $EVAL_STEPS \
    --train-batch-size $EVAL_BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --save-checkpoints-steps $SAVE_CHECKPOINTS_STEP \
    --keep-checkpoint-max $KEEP_CHECKPOINT_MAX \
    --first-layer-size $FIRST_LAYER_SIZE \
    --num-layers $NUM_LAYERS \
    --dnn-optimizer $DNN_OPTIMIZER \
    --linear-optimizer $LINEAR_OPTIMIZER \
    --include_prediction_class $INCLUDE_PREDICTION_CLASS \
    --probability_output_key $PROBABILITY_OUTPUT_KEY \
    --prediction_output_key $PREDICTION_OUTPUT_KEY
```

##### 3.2.2: Train on the Cloud

```bash
gcloud ai-platform jobs submit training $JOB_NAME \
    --stream-logs \
    --job-dir $OUTPUT_PATH \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier STANDARD_1 \
    -- \
    --estimator_type $ESTIMATOR_TYPE \
    --train-data $TRAIN_DATA \
    --eval-data $EVAL_DATA \
    --transform-dir $TRANSFORM_DIR \
    --schema-file $SCHEMA \
    --train-steps $TRAIN_STEPS \
    --eval-steps $TRAIN_BATCH_SIZE \
    --eval-batch-size $EVAL_STEPS \
    --train-batch-size $EVAL_BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --save-checkpoints-steps $SAVE_CHECKPOINTS_STEP \
    --keep-checkpoint-max $KEEP_CHECKPOINT_MAX \
    --first-layer-size $FIRST_LAYER_SIZE \
    --num-layers $NUM_LAYERS \
    --dnn-optimizer $DNN_OPTIMIZER \
    --linear-optimizer $LINEAR_OPTIMIZER \
    --include_prediction_class $INCLUDE_PREDICTION_CLASS \
    --probability_output_key $PROBABILITY_OUTPUT_KEY \
    --prediction_output_key $PREDICTION_OUTPUT_KEY
```

Refer to
[this page](https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training)
for more details on training model on Google AI platform.

#### 3.3: Deploy the model

Model can be deployed manually using [Cloud SDK](https://cloud.google.com/sdk/)
or as a part of Driblet setup.

###### 3.3.1: Manual deployment

In order to deploy saved model on
[AI Platform](https://cloud.google.com/ai-platform/), first you need to create a
model instance on the Cloud and then deploy. For further details, check
[this help page](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models).

You can deploy either using `gcloud` command or via Web UI. Above link describes
how to do both. Following is snippet how to do it using `gcloud` command.

First, setup environment variables:

```bash
MODEL_NAME=<model_name>
REGION=<asia-northeast1> # Or another region
MODEL_VERSION=<model_version>
MODEL_DIR=<path_to_saved_model>
```

DESCRIPTION:

*   `MODEL_NAME`: Name of the model.
*   `REGION`: Cloud region, refer to
    [available regions](https://cloud.google.com/ml-engine/docs/tensorflow/regions).
*   `MODEL_VERSION`: Version of your model.
*   `MODEL_DIR`: Path to saved model (local directory or Cloud Storage path).

Create a model:

```bash
gcloud ai-platform models create ${MODEL_NAME} \
    --regions $REGION
```

Deploy model:

```bash
# Deploy model with version specified above
gcloud ai-platform versions create ${MODEL_VERSION} \
    --model ${MODEL_NAME} \
    --origin ${MODEL_DIR}
```

##### 3.3.2: Programmatic deployment

Will be available soon as a part of Cloud Utils module.
