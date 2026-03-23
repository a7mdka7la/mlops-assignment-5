# Assignment 5: MLOps Pipeline with GitHub Actions

This project implements a multi-job CI/CD pipeline for ML model validation and deployment.

## Structure

```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml      # GitHub Actions workflow with validate & deploy jobs
├── train.py                  # Training script with MLflow logging
├── check_threshold.py        # Script to check accuracy threshold (0.85)
├── Dockerfile                # Container definition with ARG RUN_ID
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Pipeline Overview

### Job 1: Validate
1. **Pulls Data**: Uses `dvc pull` to get training dataset
2. **Trains**: Runs `python train.py`
3. **Observes**: Logs run to MLflow Tracking Server (uses secret `MLFLOW_TRACKING_URI`)
4. **Exports**: Creates `model_info.txt` with MLflow Run ID
5. **Persists**: Uploads `model_info.txt` as artifact using `actions/upload-artifact@v4`

### Job 2: Deploy
1. **Downloads**: Retrieves `model_info.txt` using `actions/download-artifact@v4`
2. **Logic**: Runs `check_threshold.py` to verify accuracy ≥ 0.85
3. **Halts**: Pipeline fails if accuracy is below threshold
4. **Containerizes**: Runs mock Docker build on success

## Requirements

- Python 3.10
- MLflow tracking server (set via `MLFLOW_TRACKING_URI` secret)
- DVC (optional, for data versioning)

## Setup

1. Set the `MLFLOW_TRACKING_URI` secret in your GitHub repository:
   - Go to Settings → Secrets and variables → Actions
   - Add `MLFLOW_TRACKING_URI` with your MLflow server URL

2. Ensure your MLflow server is accessible from GitHub Actions runners

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set MLflow URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Start MLflow server (in another terminal)
mlflow server --host 0.0.0.0 --port 5000

# Train model
python train.py

# Check threshold
python check_threshold.py

# Build Docker image (optional)
docker build --build-arg RUN_ID=$(cat model_info.txt) -t model-deploy .
```

## Notes

- The `train.py` script randomly varies hyperparameters, so accuracy may vary between runs
- To ensure consistent testing, you can modify `train.py` to use fixed hyperparameters
- The Dockerfile accepts `RUN_ID` as a build argument and simulates model download
