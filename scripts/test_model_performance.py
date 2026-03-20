# scripts/test_model_performance.py
#
# Pytest suite – Model performance quality gate.
#
# Evaluates the latest model version in the MLflow "Staging" stage against
# the preprocessed holdout test set and asserts that each metric meets the
# minimum acceptable threshold.  This test acts as a quality gate in the
# CI/CD pipeline: if any metric falls below the threshold the pipeline is
# blocked and the model is not promoted to Production.
#
# Minimum thresholds (configurable below):
#   - Accuracy  ≥ 0.40
#   - Precision ≥ 0.40  (weighted, zero_division=1)
#   - Recall    ≥ 0.40  (weighted, zero_division=1)
#   - F1 score  ≥ 0.40  (weighted, zero_division=1)
#
# The test is parametrised so that paths and model names can be changed
# without modifying the test logic.

import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

# --------------------------------------------------
# Connect to the remote MLflow tracking server
# --------------------------------------------------
mlflow.set_tracking_uri("http://34.224.212.114:8000/")


# --------------------------------------------------
# Test: Model performance on holdout data
# --------------------------------------------------
@pytest.mark.parametrize("model_name, stage, holdout_data_path, vectorizer_path", [
    (
        "yt_chrome_plugin_model",
        "staging",
        "data/interim/test_processed.csv",
        "tfidf_vectorizer.pkl",
    ),
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):
    """Verify that the Staging model meets the minimum performance thresholds.

    Steps:
        1. Load the latest model version from the ``stage`` in MLflow.
        2. Load the TF-IDF vectorizer from ``vectorizer_path``.
        3. Load and preprocess the holdout test dataset.
        4. Compute accuracy, precision, recall, and F1 on the holdout set.
        5. Assert that each metric is at or above the defined threshold.

    Args:
        model_name (str): Registered model name in MLflow Model Registry.
        stage (str): Model lifecycle stage to query (e.g. "staging").
        holdout_data_path (str): Path to the preprocessed holdout CSV file.
        vectorizer_path (str): Path to the pickled TF-IDF vectorizer artifact.
    """
    try:
        # --------------------------------------------------
        # Step 1: Load the model from MLflow Staging
        # --------------------------------------------------
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        latest_version = latest_version_info[0].version if latest_version_info else None

        assert latest_version is not None, (
            f"No model found in the '{stage}' stage for '{model_name}'"
        )

        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # --------------------------------------------------
        # Step 2: Load the TF-IDF vectorizer saved during training
        # --------------------------------------------------
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # --------------------------------------------------
        # Step 3: Load and prepare the holdout test dataset
        # --------------------------------------------------
        holdout_data = pd.read_csv(holdout_data_path)

        # The last column is the target label; all preceding columns are features
        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze()
        y_holdout = holdout_data.iloc[:, -1]

        # Replace NaN values in text with empty strings to avoid vectorizer errors
        X_holdout_raw = X_holdout_raw.fillna("")

        # Transform raw text into TF-IDF feature vectors and build a named DataFrame
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(
            X_holdout_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # --------------------------------------------------
        # Step 4: Run inference on the holdout set
        # --------------------------------------------------
        y_pred_new = model.predict(X_holdout_tfidf_df)

        # --------------------------------------------------
        # Step 5: Calculate and assert performance metrics
        # --------------------------------------------------
        accuracy_new  = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new    = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new        = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        # Minimum acceptable thresholds – update these as the model improves
        expected_accuracy  = 0.40
        expected_precision = 0.40
        expected_recall    = 0.40
        expected_f1        = 0.40

        assert accuracy_new  >= expected_accuracy,  (
            f"Accuracy below threshold: expected >= {expected_accuracy}, got {accuracy_new:.4f}"
        )
        assert precision_new >= expected_precision, (
            f"Precision below threshold: expected >= {expected_precision}, got {precision_new:.4f}"
        )
        assert recall_new    >= expected_recall,    (
            f"Recall below threshold: expected >= {expected_recall}, got {recall_new:.4f}"
        )
        assert f1_new        >= expected_f1,        (
            f"F1 score below threshold: expected >= {expected_f1}, got {f1_new:.4f}"
        )

        print(f"Performance test passed for model '{model_name}' version {latest_version}")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")
