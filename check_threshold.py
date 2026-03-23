"""
Script to check if model accuracy meets the threshold.
Exits with error code 1 if accuracy < 0.85.
"""
import os
import sys
import mlflow

# Threshold for deployment
ACCURACY_THRESHOLD = 0.85


def check_threshold():
    """Check if model accuracy meets threshold."""
    # Set MLflow tracking URI from environment variable
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Read run ID from file
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("ERROR: model_info.txt not found!")
        sys.exit(1)

    print(f"Checking accuracy for Run ID: {run_id}")

    # Get run metrics from MLflow
    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy", 0.0)
    except Exception as e:
        print(f"ERROR: Failed to fetch run from MLflow: {e}")
        sys.exit(1)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Required threshold: {ACCURACY_THRESHOLD}")

    if accuracy < ACCURACY_THRESHOLD:
        print(f"ERROR: Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}")
        print("Deployment halted!")
        sys.exit(1)
    else:
        print(f"SUCCESS: Accuracy {accuracy:.4f} meets threshold {ACCURACY_THRESHOLD}")
        print("Proceeding with deployment...")
        sys.exit(0)


if __name__ == "__main__":
    check_threshold()
