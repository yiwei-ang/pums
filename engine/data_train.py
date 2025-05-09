import os
import json
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline

from xgboost import plot_importance
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Prepare data from Snowflake for modeling or inference.")
parser.add_argument("--input_file", required=True, default="working_dir/data_train.parquet", help="Locally parquet that stores data for training")
args = parser.parse_args()
# Example:
# python engine/data_train.py --input_file working_dir/data_train.parquet

# Set experiment (creates if it doesn't exist)
mlflow.set_experiment("food_stamp_prediction")

# Turn on autologging for XGBoost
mlflow.xgboost.autolog()

# Start run
with mlflow.start_run(run_name="xgb_weighted_fs_model") as run:
    df = pd.read_parquet(os.path.join(args.input_path))
    run_id = run.info.run_id
    experiment = mlflow.get_experiment_by_name("food_stamp_prediction")

    # Exclude non-feature columns
    exclusions = ["ELEP", "FULP", "GASP", "INSP", "MHP", "MRGP", "RNTP", "SMP", "SMOCP", "TAXAMT", "VALP", "WATP"] + \
                 ['FS', 'RT', 'SERIALNO', 'NAICSP', 'SOCP', 'ADJINC', 'ADJHSG', 'WGTP', 'PWGTP'] + \
                 ['WGTP' + str(i) for i in range(81)] + ['PWGTP' + str(i) for i in range(81)] + \
                 ['VACS', 'VACDUR', 'VACOTH']
    features = [col for col in df.columns if col not in exclusions]

    # Prepare data
    X = df[features]
    y = df["FS"]
    weight = df['WGTP']
    X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(
        X, y, weight, test_size=0.1, random_state=42
    )

    # Define pipeline with scaler and XGB
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        ))
    ])

    # Fit with sample weight (note: only XGBoost step gets sample_weight)
    model.fit(X_train, y_train, xgb__sample_weight=weight_train)

    # Log entire pipeline
    mlflow.sklearn.log_model(model, "model", input_example=X_train.sample(5))

    # --- Register the model ---
    model_uri = f"runs:/{run_id}/model"
    model_name = "snap_xgb_classifier"  # you can rename this

    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except:
        client.create_registered_model(model_name)

    # Register the model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

    # Save required info to config.json
    train_config = {
        "features": features,
        "label": ["FS"],
        "weight": ['WGTP']
    }

    with open("train.config", "w") as f:
        json.dump(train_config, f, indent=4)
    mlflow.log_artifact("train.config")
    os.remove("train.config")

    # Predict proba
    y_proba = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba, sample_weight=weight_test)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    # Log custom threshold
    mlflow.log_param("custom_threshold", round(best_threshold, 4))
    mlflow.log_metric("best_f1_weighted", f1_scores[best_idx])

    # Predictions with default and best threshold
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_custom = (y_proba >= best_threshold).astype(int)

    # Classification reports
    report_default = classification_report(y_test, y_pred_default, sample_weight=weight_test)
    report_custom = classification_report(y_test, y_pred_custom, sample_weight=weight_test)

    print("\n Default Threshold (0.5):")
    print(report_default)
    print("\n Best Threshold (%.2f):" % best_threshold)
    print(report_custom)

    # Save both reports to a text file
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("=== Default Threshold (0.5) ===\n")
        f.write(report_default + "\n\n")
        f.write(f"=== Best Threshold ({best_threshold:.2f}) ===\n")
        f.write(report_custom)

    # Log the report as artifact
    mlflow.log_artifact(report_path)

    # Clean up
    os.remove(report_path)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="PR Curve", lw=2)
    plt.scatter(recall[best_idx], precision[best_idx], color="red", label=f"Best F1 (Thresh={best_threshold:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    # Save the plot
    pr_curve_path = "pr_curve.png"
    plt.savefig(pr_curve_path)
    mlflow.log_artifact(pr_curve_path)
    plt.close()

    # Clean up
    os.remove(pr_curve_path)

    # Plot top 20 by gain
    xgb_model = model.named_steps["xgb"]
    xgb_model.get_booster().feature_names = X_train.columns.tolist()

    plt.figure(figsize=(10, 8))
    plot_importance(xgb_model, max_num_features=20, importance_type='gain')
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()

    # Save & log
    plt.savefig("feature_importance_gain.png")
    mlflow.log_artifact("feature_importance_gain.png")
    plt.close()
    os.remove("feature_importance_gain.png")

    # Print useful info
    print("\n--- MLflow Run Info ---")
    print(f"Run ID        : {run_id}")
    print(f"Experiment ID : {experiment.experiment_id}")
    print(f"Experiment    : {experiment.name}")
    print(f"MLflow UI     : http://localhost:5000/#/experiments/{experiment.experiment_id}/runs/{run_id}")
