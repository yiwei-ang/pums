# SNAP Inference Pipeline using Dagster

This Dagster pipeline (`snap_inference_job`) automates the following stages:

1. **Data Preparation** – Joins housing and person-level ACS PUMS data from Snowflake.
2. **Model Inference** – Loads a pre-registered MLflow model to score SNAP (food stamp) eligibility.
3. **Upload to S3** – Saves the predictions to an S3 bucket for downstream use.

## Pre-requisite
1. Configure environment variable to include
2. Have aws configure ready.

## Project Structure
```angular2html
pums/
├── dagster/
│ ├── job.py # Dagster job + ops
│ └── config.yaml # Configuration for job launch
├── engine/
│ ├── data_preparation.py # SQL-based feature engineering
│ ├── data_inference.py # Runs model inference
│ └── utils/
│ └── s3_uploader.py # Uploads scored parquet to S3****
```
## Running the Job
1. Make sure your Python environment has all required packages:

2. Set your environment to access Dagster UI (http://127.0.0.1:3000/):
```bash
export DAGSTER_HOME=./dagster_home
dagster dev -f dagster/job.py
```

3. Trigger the job with config:
```
dagster job launch \
  -f dagster/job.py \
  -j snap_inference_job \
  -c dagster/config.yaml
```
or in Dagster UI -> Jobs -> Launch Pad -> Execute.


4. View S3 file: [S3 link](https://us-west-2.console.aws.amazon.com/s3/buckets/food-stamp-prediction?region=us-west-2&bucketType=general&prefix=snap_outputs/tx/&showversions=false)