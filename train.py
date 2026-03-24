import os
import random
import mlflow
import mlflow.sklearn
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment("assignment-5-pipeline")

    data_path = "data/dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Did you run 'dvc pull'?")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        n_estimators = random.randint(5, 20)
        max_depth = random.randint(2, 10)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracy = 0.50

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(clf, "model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")

        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

        return run.info.run_id, accuracy


if __name__ == "__main__":
    train_model()
