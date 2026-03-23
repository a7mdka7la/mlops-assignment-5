# Dockerfile for model deployment
FROM python:3.10-slim

# Accept RUN_ID as build argument
ARG RUN_ID

# Set environment variable from ARG
ENV MODEL_RUN_ID=${RUN_ID}

# Install dependencies
RUN pip install --no-cache-dir mlflow scikit-learn

# Create app directory
WORKDIR /app

# Copy scripts
COPY train.py check_threshold.py ./

# Simulate downloading the model using the RUN_ID
# In a real scenario, this would download from MLflow model registry
RUN echo "Downloading model for Run ID: ${MODEL_RUN_ID}" && \
    echo "Model download completed (simulated)"

# Default command
CMD ["sh", "-c", "echo Container ready. Model Run ID: ${MODEL_RUN_ID}"]
