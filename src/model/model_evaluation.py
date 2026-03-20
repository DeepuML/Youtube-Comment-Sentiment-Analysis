# src/model/model_evaluation.py
#
# Model evaluation stage for the YouTube Comment Sentiment Analysis pipeline.
#
# This module is the fourth stage in the DVC-orchestrated ML pipeline.  It:
#   1. Loads the trained LightGBM model and the fitted TF-IDF vectorizer.
#   2. Evaluates the model on the preprocessed holdout test set.
#   3. Logs all metrics, artifacts, and a model signature to MLflow.
#   4. Saves the MLflow run ID and model path to experiment_info.json so the
#      next pipeline stage (model registration) can locate the correct run.
#
# Output artifacts (logged to MLflow):
#   - Classification report metrics per class
#   - Confusion matrix visualisation (PNG)
#   - Trained model with inferred signature and input example
#   - TF-IDF vectorizer pickle file
#
# Usage (via DVC):
#   dvc repro model_evaluation
#
# Usage (standalone):
#   python src/model/model_evaluation.py

import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# --------------------------------------------------
# Logging configuration for model evaluation stage
# --------------------------------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Console handler outputs all DEBUG and above messages to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler captures only ERROR and above messages for post-run inspection
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

# Consistent log format: timestamp - logger name - log level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --------------------------------------------------
# Function: Load CSV data into a DataFrame
# --------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame and replace NaN values.

    Args:
        file_path (str): Path to the CSV file to load.

    Returns:
        pd.DataFrame: Loaded DataFrame with NaN values replaced by empty strings.

    Raises:
        Exception: Re-raises any exception encountered during file loading.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Replace NaN with empty strings to avoid vectorizer errors
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


# --------------------------------------------------
# Function: Load the trained model from a pickle file
# --------------------------------------------------
def load_model(model_path: str):
    """Deserialise and return the trained LightGBM model from a pickle file.

    Args:
        model_path (str): Path to the serialised model file (.pkl).

    Returns:
        lgb.LGBMClassifier: The deserialised LightGBM model object.

    Raises:
        Exception: Re-raises any exception encountered during file loading.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


# --------------------------------------------------
# Function: Load the fitted TF-IDF vectorizer
# --------------------------------------------------
def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Deserialise and return the fitted TF-IDF vectorizer from a pickle file.

    The vectorizer must be the same instance that was fitted during the model
    building stage so that the feature space matches the model's expectations.

    Args:
        vectorizer_path (str): Path to the serialised vectorizer file (.pkl).

    Returns:
        TfidfVectorizer: The fitted TF-IDF vectorizer.

    Raises:
        Exception: Re-raises any exception encountered during file loading.
    """
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


# --------------------------------------------------
# Function: Load pipeline parameters from YAML
# --------------------------------------------------
def load_params(params_path: str) -> dict:
    """Load pipeline hyperparameters and configuration from a YAML file.

    Args:
        params_path (str): Path to the params.yaml configuration file.

    Returns:
        dict: Parsed YAML content as a Python dictionary.

    Raises:
        Exception: Re-raises any exception encountered during file loading.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


# --------------------------------------------------
# Function: Evaluate the model on test data
# --------------------------------------------------
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Run inference on the test set and compute classification metrics.

    Args:
        model: Trained classifier with a ``predict`` method.
        X_test (np.ndarray): TF-IDF feature matrix for the test set.
        y_test (np.ndarray): True sentiment labels for the test set.

    Returns:
        tuple: A (report, cm) pair where:
            - report (dict): Per-class precision, recall, F1, and support
              as returned by ``sklearn.metrics.classification_report``.
            - cm (np.ndarray): Confusion matrix.

    Raises:
        Exception: Re-raises any exception encountered during evaluation.
    """
    try:
        # Generate predictions for the full test set
        y_pred = model.predict(X_test)

        # Compute per-class and aggregate classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        # Compute the confusion matrix for visualisation
        cm = confusion_matrix(y_test, y_pred)

        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


# --------------------------------------------------
# Function: Log confusion matrix to MLflow as an artifact
# --------------------------------------------------
def log_confusion_matrix(cm, dataset_name):
    """Render the confusion matrix as a heatmap and log it to MLflow.

    The chart is saved to a local PNG file first (required by MLflow's
    artifact logging API) and then uploaded to the active run.

    Args:
        cm (np.ndarray): Confusion matrix to visualise.
        dataset_name (str): Label used in the chart title and file name
                            (e.g. "Test Data").
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Persist the figure locally then upload it to the current MLflow run
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


# --------------------------------------------------
# Function: Persist run metadata for downstream stages
# --------------------------------------------------
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the MLflow run ID and logged model path to a JSON file.

    The downstream model registration stage reads this file to locate the
    correct run and model artifact without having to query MLflow.

    Args:
        run_id (str): MLflow run ID of the evaluation run.
        model_path (str): Relative path of the logged model artifact within
                          the MLflow run (e.g. "lgbm_model").
        file_path (str): Destination path for the output JSON file
                         (e.g. "experiment_info.json").

    Raises:
        Exception: Re-raises any exception encountered while writing the file.
    """
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


# --------------------------------------------------
# Main function: End-to-end model evaluation pipeline
# --------------------------------------------------
def main():
    """Execute the full model evaluation workflow and log results to MLflow."""
    # Connect to the remote MLflow tracking server
    mlflow.set_tracking_uri("http://34.224.212.114:8000/")

    # Target experiment – all evaluation runs are grouped here
    mlflow.set_experiment('dvc-pipeline-runs1')

    with mlflow.start_run() as run:
        try:
            # --------------------------------------------------
            # Load pipeline hyperparameters and log them to MLflow
            # --------------------------------------------------
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log every top-level parameter section to the MLflow run for reproducibility
            for key, value in params.items():
                mlflow.log_param(key, value)

            # --------------------------------------------------
            # Load the trained model and fitted vectorizer artifacts
            # --------------------------------------------------
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # --------------------------------------------------
            # Prepare test data for evaluation and signature inference
            # --------------------------------------------------
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Transform the cleaned comment text into TF-IDF feature vectors
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Use the first 5 rows as an input example for the MLflow model signature
            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5],
                columns=vectorizer.get_feature_names_out()
            )

            # Infer the model signature (input schema → output schema) from sample data
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # --------------------------------------------------
            # Log the model with its signature and an input example to MLflow
            # --------------------------------------------------
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,       # Enables MLflow to validate future inputs
                input_example=input_example  # Stored alongside the model for documentation
            )

            # Persist run ID and model path so the registration stage can find this run
            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the TF-IDF vectorizer as a separate artifact for reproducibility
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # --------------------------------------------------
            # Evaluate the model and log classification metrics
            # --------------------------------------------------
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log per-class precision, recall, and F1 for the test split
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall":    metrics['recall'],
                        f"test_{label}_f1-score":  metrics['f1-score']
                    })

            # Log the confusion matrix visualisation as a PNG artifact
            log_confusion_matrix(cm, "Test Data")

            # --------------------------------------------------
            # Tag the run with descriptive metadata for easier discovery
            # --------------------------------------------------
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == '__main__':
    main()