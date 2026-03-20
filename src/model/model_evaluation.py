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

# Console handler for debug-level logs
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler to store error-level logs in a file
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

# Log formatting: timestamp - logger name - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add both console and file handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --------------------------------------------------
# Function: Load CSV data into DataFrame
# --------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values with empty strings
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


# --------------------------------------------------
# Function: Load trained model from pickle file
# --------------------------------------------------
def load_model(model_path: str):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


# --------------------------------------------------
# Function: Load TF-IDF vectorizer from pickle file
# --------------------------------------------------
def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer from a pickle file."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


# --------------------------------------------------
# Function: Load parameters from YAML file
# --------------------------------------------------
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


# --------------------------------------------------
# Function: Evaluate model and return metrics
# --------------------------------------------------
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return classification report and confusion matrix."""
    try:
        # Predict labels for the test set
        y_pred = model.predict(X_test)
        # Generate per-class precision, recall, and F1-score
        report = classification_report(y_test, y_pred, output_dict=True)
        # Build the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


# --------------------------------------------------
# Function: Log confusion matrix as MLflow artifact
# --------------------------------------------------
def log_confusion_matrix(cm, dataset_name):
    """Plot the confusion matrix, save it as a PNG, and log it to MLflow."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


# --------------------------------------------------
# Function: Persist MLflow run info to JSON
# --------------------------------------------------
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the MLflow run ID and model path to a JSON file for downstream use."""
    try:
        # Build the info dictionary to persist
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Write the dictionary as a formatted JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


# --------------------------------------------------
# Main function: End-to-end model evaluation process
# --------------------------------------------------
def main():
    # Point MLflow client at the remote tracking server
    mlflow.set_tracking_uri("http://34.224.212.114:8000/")

    mlflow.set_experiment('dvc-pipeline-runs1')

    with mlflow.start_run() as run:
        try:
            # Determine project root directory (two levels up from this script)
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

            # Load pipeline parameters and log them to MLflow
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load trained model and TF-IDF vectorizer from disk
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load preprocessed test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Transform test comments using the trained vectorizer
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Build an input example DataFrame for model signature inference
            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5],
                columns=vectorizer.get_feature_names_out()
            )

            # Infer input/output schema for the MLflow model signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Log model to MLflow with its signature and an input example
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )

            # Persist the run ID and model path for the registration step
            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the vectorizer pickle as a run artifact
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model performance on the test set
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log per-class precision, recall, and F1-score to MLflow
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log the confusion matrix image as an artifact
            log_confusion_matrix(cm, "Test Data")

            # Tag the run with descriptive metadata
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