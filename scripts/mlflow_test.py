# scripts/mlflow_test.py
#
# MLflow connectivity smoke test for the YouTube Comment Sentiment Analysis project.
#
# This script performs a minimal end-to-end connectivity check against the remote
# MLflow tracking server by logging dummy parameters and metrics inside a new run.
#
# Run this script manually whenever you need to verify that:
#   - The MLflow tracking server is reachable at the configured URI.
#   - The service account / credentials used in CI/CD have write access.
#
# Usage:
#   python scripts/mlflow_test.py

import mlflow
import random

# --------------------------------------------------
# Connect to the remote MLflow tracking server
# --------------------------------------------------
mlflow.set_tracking_uri("http://34.224.212.114:8000/")

# --------------------------------------------------
# Log dummy parameters and metrics to verify connectivity
# --------------------------------------------------
with mlflow.start_run():
    # Log arbitrary hyperparameter values to confirm param logging works
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log arbitrary metric values to confirm metric logging works
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")