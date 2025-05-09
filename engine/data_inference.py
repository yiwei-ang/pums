import argparse
import json
import pandas as pd
import mlflow


# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Prepare data from Snowflake for modeling or inference.")
parser.add_argument("--input_file", required=True, default="working_dir/data.parquet", help="Locally parquet that stores data for inference")
parser.add_argument("--output_file", required=True, default="working_dir/scored_data.parquet", help="Locally parquet that stores data for inference")
args = parser.parse_args()
# Example:
# python engine/data_inference.py --input_file working_dir/data.parquet --output_file working_dir/scored_data.parquet

# --- Config ---
model_name = "snap_xgb_classifier"
DATA_PATH = "working_dir/data.parquet"

# --- Load model from registry ---
logged_model = f"models:/{model_name}/5"
loaded_model = mlflow.sklearn.load_model(logged_model)

# --- Get custom threshold from run params ---
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(model_name)[0]
run_id = latest_version.run_id
run = client.get_run(run_id)
custom_threshold = float(run.data.params.get("custom_threshold", 0.5))
print(f"Using custom threshold: {custom_threshold}")
print(latest_version)

# --- Load config ---
experiment_id = run.info.experiment_id
config_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/train.config"
with open(config_uri, "r") as f:
    config = json.load(f)

# --- Load and predict ---
df = pd.read_parquet(DATA_PATH)
X = df[config['features']]
proba = loaded_model.predict_proba(X)[:, 1]
df["proba_FS"] = proba
df["pred_FS"] = (proba >= 0.5).astype(int)
df["pred_FS_custom"] = (proba >= custom_threshold).astype(int)

# --- Save
df[["SERIALNO", "pred_FS", "pred_FS_custom", "FS"]].to_parquet(args.output_file, index=False)
print("Scored data saved.")
