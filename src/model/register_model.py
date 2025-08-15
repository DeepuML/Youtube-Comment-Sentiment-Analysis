# ----------------------------------------------------------
# register_model.py
# Registers a trained ML model to MLflow Model Registry
# and attempts to transition it to the "Staging" stage.
# ----------------------------------------------------------

import json
import mlflow
import logging
import requests
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

# -------------------------------
# Configure MLflow Tracking URI
# -------------------------------
mlflow.set_tracking_uri("http://52.204.155.208:8000/")

# -------------------------------
# Logging setup
# -------------------------------
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------------------
# Load model info
# -------------------------------
def load_model_info(file_path: str) -> dict:
    """
    Load the model information (run_id and model_path) from a JSON file.
    """
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logger.debug('Model info loaded from %s', file_path)
    return model_info

# -------------------------------
# Register model
# -------------------------------
def register_model(model_name: str, model_info: dict):
    """
    Register the trained model in MLflow Model Registry (non-blocking).
    """
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    client = MlflowClient()

    # Create a new version without waiting for artifacts check
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=model_info['run_id']
    )
    logger.debug(f"Model {model_name} version {model_version.version} registered (no wait).")

    # Attempt stage transition
    try:
        logger.debug("Attempting to transition model to Staging...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version} transitioned to Staging.")
    except (RestException, requests.exceptions.RequestException) as e:
        logger.error(f"Stage transition failed: {e}. Skipping stage update.")

# -------------------------------
# Main function
# -------------------------------
def main():
    try:
        model_info = load_model_info('experiment_info.json')
        register_model("yt_chrome_plugin_model", model_info)
    except Exception as e:
        logger.error(f'Failed to complete the model registration process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
