# scripts/promote_model.py
#
# Model promotion utility for the YouTube Comment Sentiment Analysis project.
#
# This script is executed as the final step of the CI/CD pipeline after all
# quality-gate tests have passed.  It performs the following operations:
#
#   1. Archives any model version currently in the "Production" stage so
#      that only one model version is active in Production at a time.
#   2. Promotes the latest "Staging" model version to "Production".
#
# Usage:
#   python scripts/promote_model.py
#
# The MLflow tracking server must be reachable at the configured URI
# and the model name must already exist in the MLflow Model Registry.

import os
import mlflow


# --------------------------------------------------
# Model promotion function
# --------------------------------------------------
def promote_model():
    """Archive the current Production model and promote the latest Staging model.

    Connects to the remote MLflow tracking server, retrieves the most recent
    model version in the "Staging" stage, archives every version currently in
    "Production", and then transitions the Staging version to "Production".

    Raises:
        IndexError: If no model version is found in the "Staging" stage.
        mlflow.exceptions.MlflowException: If the MLflow API call fails.
    """
    # Connect to the remote MLflow tracking server
    mlflow.set_tracking_uri("http://34.224.212.114:8000/")

    client = mlflow.MlflowClient()

    model_name = "yt_chrome_plugin_model"

    # Retrieve the latest version currently sitting in the Staging stage
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive all versions currently in Production before promoting the new one.
    # This ensures that only a single version is active in Production at any time.
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the validated Staging model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")


# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == "__main__":
    promote_model()