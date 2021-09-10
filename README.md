# Driblet - ML pipeline orchestration framework

*   [Overview](#overview)
*   [Step 1: Local environment setup](#step-1-local-environment-setup)
*   [Step 2: Cloud environment setup](#step-2-cloud-environment-setup)
*   [Step 3: Configure Airflow](#step-3-configure-airflow)

## Overview

Driblet is [Cloud Composer](https://cloud.google.com/composer) based pipeline to
manage dataset creation, machine learning modeling and ads activation.

Setup and deployment involves following four steps.

## Step 1: Clone the source code

1.  Select or create a
    [Google Cloud Platform project](https://console.cloud.google.com/projectcreate?).
2.  Activate
    [Cloud Shell](https://cloud.google.com/shell/docs/using-cloud-shell#starting_a_new_session).
3.  Clone Driblet repository.

    ```
    git clone https://cse.googlesource.com/solutions/driblet
    ```

## Step 2: Create service account

The service account is required to install necessary Python modules and deploy
Driblet services.

1.  Create service account following
    [this guideline](https://cloud.google.com/iam/docs/creating-managing-service-accounts)
    and grant `Editor` role. Take note on service account name, example:
    `my-service@project.iam.gserviceaccount.com`.
2.  Add `Service Account Token Creator` Role to project owner. This role enables
    impersonation of service accounts to create OAuth2 access tokens, which is
    required to run next steps. For more information refer to
    [Service accounts help page](https://cloud.google.com/iam/docs/service-accounts).

## Step 3: Setup cloud environment

Run following command to setup Cloud environment:

```
sh setup.sh
```

The will execute following six steps and take about 40 minutes to finish:

1.  Create Python virtual environment.
2.  Activate virtual environment.
3.  Install all required Python packages.
4.  Enable required Cloud APIs in GCP project.
5.  Create Cloud Composer environment.
6.  Setup Apache Airflow with ML pipelines.

## Step 4: Configure [Airflow Variables](https://airflow.apache.org/docs/stable/concepts.html#variables)

Driblet services and tasks are managed on
[Airflow UI](https://airflow.apache.org/docs/stable/ui.html). Once above steps
are done, the Airflow UI will be available to further configure the pipeline.

1.  Edit `configs/airflow_variables.json` and update all required fields
    (dataset/table names, dates, schedule etc).
2.  Visit
    [Composer environments page](http://console.cloud.google.com/composer).
3.  Open `driblet-env` Composer environment and visit `Airflow web UI` using the
    URL in present in the configuration.
4.  Goto `Admin -> Variables` and import Airflow variables using
    `airflow_variables.json` file.
