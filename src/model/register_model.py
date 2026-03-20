# src/model/register_model.py
#
# Model registration stage for the YouTube Comment Sentiment Analysis pipeline.
#
# This module is the fifth and final stage in the DVC-orchestrated ML pipeline.
# It reads the MLflow run ID and model artifact path produced by the evaluation
# stage, registers the trained model in the MLflow Model Registry under a
# versioned entry, and attempts to transition that version to "Staging" so that
# the CI/CD quality-gate tests can run against it.
#
# If the stage transition fails (e.g. MLflow server timeout), the error is
# logged and the script exits cleanly – the model version remains in "None"
# stage and can be promoted manually.
#
# Usage (via DVC):
#   dvc repro model_registration
#
# Usage (standalone):
#   python src/model/register_model.py

import json
import mlflow
import logging
import requests
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# Connect to the remote MLflow tracking server
# --------------------------------------------------
mlflow.set_tracking_uri("http://34.224.212.114:8000/")


# --------------------------------------------------
# Logging configuration for model registration stage
# --------------------------------------------------
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

# Console handler outputs all DEBUG and above messages to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler captures only ERROR and above messages for post-run inspection
file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

# Consistent log format: timestamp - logger name - log level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------------------------------------
# Function: Load experiment metadata from JSON
# --------------------------------------------------
def load_model_info(file_path: str) -> dict:
    """Load the MLflow run ID and model artifact path from a JSON file.

    This JSON file is written by the model evaluation stage and contains
    the information needed to locate the correct run in MLflow.

    Args:
        file_path (str): Path to the JSON file (e.g. "experiment_info.json").

    Returns:
        dict: Dictionary with keys ``run_id`` (str) and ``model_path`` (str).

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logger.debug('Model info loaded from %s', file_path)
    return model_info


# --------------------------------------------------
# Function: Register model in MLflow Registry
# --------------------------------------------------
def register_model(model_name: str, model_info: dict):
    """Register a trained model run in the MLflow Model Registry.

    Creates a new model version from the specified MLflow run, then attempts
    to transition that version to the "Staging" lifecycle stage.  If the
    stage transition fails due to a network or server error it is logged and
    skipped so the pipeline does not fail catastrophically.

    Args:
        model_name (str): Name to use in the MLflow Model Registry.
                          The registry entry is created if it does not exist.
        model_info (dict): Dictionary containing:
            - ``run_id`` (str): The MLflow run ID that produced the model.
            - ``model_path`` (str): Relative path of the model artifact
              within that run (e.g. "lgbm_model").

    Raises:
        mlflow.exceptions.MlflowException: If model version creation fails.
    """
    # Build the MLflow model URI from the run ID and artifact path
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    client = MlflowClient()

    # Register a new version of the model without waiting for artifact validation
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=model_info['run_id']
    )
    logger.debug(f"Model {model_name} version {model_version.version} registered (no wait).")

    # Attempt to transition the new version to Staging for CI/CD quality-gate tests
    try:
        logger.debug("Attempting to transition model to Staging...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version} transitioned to Staging.")
    except (RestException, requests.exceptions.RequestException) as e:
        # Log the failure but do not raise -- the model is registered even if
        # the stage transition could not be completed
        logger.error(f"Stage transition failed: {e}. Skipping stage update.")


# --------------------------------------------------
# Main function: Run the model registration workflow
# --------------------------------------------------
def main():
    """Load experiment metadata and register the model in MLflow Registry."""
    try:
        # Read the run ID and model path written by the evaluation stage
        model_info = load_model_info('experiment_info.json')

        # Register the model and promote it to Staging
        register_model("yt_chrome_plugin_model", model_info)
    except Exception as e:
        logger.error(f'Failed to complete the model registration process: {e}')
        print(f"Error: {e}")


# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == '__main__':
    main()
