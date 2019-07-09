# Model Training Guide

Model training and deploying involves 3 steps:

1.  [Environment setup](#step-1-environment-setup)
2.  [Train the model](#step-2-train-the-model)
3.  [Deploy the model](#step-3-deploy-the-model)

Model provided in this directory is
[DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier).
Refer to
[Wide & Deep Learning: Better Together with TensorFlow](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
for more details.

NOTE: Before you start training, make sure all features are correctly configured
in `workflow/dags/tasks/preprocess/features_config.py`. Refer to step
[Configure features](../../driblet/README.md#1-configure-features).

## Step 1: Environment setup

Before starting training following env variables need to be set:

```bash
MODEL_DIR=<model_dir>
TRAIN_DATA=<full_path_to_train_data>
EVAL_DATA=<full_path_to_eval_data>
TRANSFORM_DIR=<path_to_transfomer_dir>
SCHEMA=<path_to_schema_proto>
TRAIN_STEPS=<train_steps>
TRAIN_BATCH_SIZE=<train_batch_size>
EVAL_STEPS=<eval_steps>
EVAL_BATCH_SIZE=<eval_batch_size>
NUM_EPOCHS=<num_epochs>
SAVE_CHECKPOINTS_STEP=<save_checkpoints_step>
KEEP_CHECKPOINT_MAX=<keep_checkpoint_max>
FIRST_LAYER_SIZE=<first_layer_size>
NUM_LAYERS=<num_layers>
DNN_OPTIMIZER=<dnn_optimizer>
LINEAR_OPTIMIZER=<linear_optimizer>
```

DESCRIPTION:

*   `MODEL_DIR`: Local path or path in Cloud Storage to store checkpoints and
    save the trained model.
*   `TRAIN_DATA`: Local path or path in Cloud Storage holding train dataset.
*   `EVAL_DATA`: Local path or path in Cloud Storage holding eval dataset.
*   `TRANSFORM_DIR`: Local path or path in Cloud Storage that holds the model.
    saved during data transformation
*   `SCHEMA`: Local path or path in Cloud Storage to `schema.pbtxt` file.
    generated during data transformation
*   `TRAIN_STEPS`: Count of steps to run the training job for.
*   `TRAIN_BATCH_SIZE`: Train batch size.
*   `EVAL_STEPS`: Number of steps to run evaluation for at each checkpoint.
*   `EVAL_BATCH_SIZE`: Eval batch size.
*   `NUM_EPOCHS`: Number of epochs.
*   `SAVE_CHECKPOINTS_STEP`: Save checkpoints every this many steps.
*   `KEEP_CHECKPOINT_MAX`: The maximum number of recent checkpoint files to
    keep.
*   `FIRST_LAYER_SIZE`: Size of the first layer.
*   `NUM_LAYERS`: Number of layers..
*   `DNN_OPTIMIZER`: Optimizer for DNN model.
*   `LINEAR_OPTIMIZER`: Optimizer for linear model.

EXAMPLE:

```bash
export MODEL_DIR=~/sample_output/model
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
```

## Step 2: Train the model

Model can be trained locally or on the
[Cloud AI Platform](https://cloud.google.com/ml-engine/docs/tensorflow/training-overview).

### Option 1: Train locally

From root directory of the project, run following:

*NOTE: hyperparameters are set arbitrarily.*

```bash
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
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
    --linear-optimizer $LINEAR_OPTIMIZER
```

### Option 2: Train on the Cloud

```bash
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --job-dir $OUTPUT_PATH \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier STANDARD_1 \
    -- \
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
    --linear-optimizer $LINEAR_OPTIMIZER
```

Refer to
[this page](https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training)
for more details on training model on Google AI platform.

## Step 3: Deploy the model

Model can be deployed manually using [Cloud SDK](https://cloud.google.com/sdk/)
or as a part of Driblet setup.

### Option 1: Manual deployment

In order to deploy saved model on
[AI Platform](https://cloud.google.com/ai-platform/), first you need to create a
model instance on the Cloud and then deploy. For further details, check
[this help page](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models).

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
gcloud ml-engine models create ${MODEL_NAME} \
    --regions $REGION
```

Deploy model:

```bash
# Deploy model with version specified above
gcloud ml-engine versions create ${MODEL_VERSION} \
    --model ${MODEL_NAME} \
    --origin ${MODEL_DIR}
```

### Option 2: Automatic deployment

For automatic deployment as a part of Cloud environment setup, refer to
[Cloud environment setup](../README.md#step-4-cloud-services-setup).
