# scripts/test_model_signature.py
#
# Pytest suite – Model signature and inference shape test.
#
# Validates that the latest model version in the MLflow "Staging" stage can
# accept a TF-IDF–transformed input and produce predictions whose shape
# matches the number of input samples.
#
# This test guards against:
#   - Mismatches between the vectorizer feature space and the model's
#     expected input dimensionality (e.g. after a vocabulary change).
#   - Silent failures where the model returns a result of unexpected shape.
#
# The test is parametrised so that additional model/vectorizer combinations
# can be added easily in future iterations.

import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# Connect to the remote MLflow tracking server
# --------------------------------------------------
mlflow.set_tracking_uri("http://34.224.212.114:8000/")


# --------------------------------------------------
# Test: Model inference with TF-IDF vectorizer
# --------------------------------------------------
@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    """Verify that the model accepts vectorised input and returns correctly shaped output.

    Steps:
        1. Retrieve the latest model version in ``stage`` from MLflow.
        2. Load the TF-IDF vectorizer from ``vectorizer_path``.
        3. Transform a dummy comment string into a TF-IDF feature DataFrame.
        4. Run inference and assert:
           a. The input DataFrame column count matches the vectorizer vocabulary size.
           b. The number of prediction rows matches the number of input samples.

    Args:
        model_name (str): Registered model name in MLflow Model Registry.
        stage (str): Model lifecycle stage to query (e.g. "staging").
        vectorizer_path (str): Path to the pickled TF-IDF vectorizer artifact.
    """
    client = MlflowClient()

    # Retrieve the latest version metadata for the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, (
        f"No model found in the '{stage}' stage for '{model_name}'"
    )

    try:
        # Load the model artifact from the MLflow Registry
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the TF-IDF vectorizer that was saved during training
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Create a minimal dummy comment to exercise the inference path
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])

        # Build a named DataFrame matching the feature space the model expects
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Run inference on the dummy input
        prediction = model.predict(input_df)

        # Assert: input feature count must equal the vectorizer vocabulary size
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), (
            "Input feature count mismatch between vectorizer and model"
        )

        # Assert: number of predictions must match number of input samples
        assert len(prediction) == input_df.shape[0], (
            "Output row count does not match the number of input samples"
        )

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
