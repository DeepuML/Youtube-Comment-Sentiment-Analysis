# scripts/test_load_model.py
#
# Pytest suite – Model loading smoke test.
#
# Verifies that the latest model version registered in the MLflow "Staging"
# stage can be loaded without errors.  This test is executed as part of the
# CI/CD pipeline immediately after a new model version has been registered
# to confirm that the artifact is intact and the MLflow Model Registry is
# accessible.
#
# The test is parametrised so that additional model names or stages can be
# added in future without duplicating test code.

import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# Connect to the remote MLflow tracking server
# --------------------------------------------------
mlflow.set_tracking_uri("http://34.224.212.114:8000/")


# --------------------------------------------------
# Test: Load latest Staging model
# --------------------------------------------------
@pytest.mark.parametrize("model_name, stage", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_load_latest_staging_model(model_name, stage):
    """Verify that the latest model version in the given stage loads successfully.

    Steps:
        1. Query the MLflow Model Registry for the latest version in ``stage``.
        2. Assert that at least one version exists in that stage.
        3. Load the model artifact via ``mlflow.pyfunc.load_model``.
        4. Assert that the returned model object is not None.

    Args:
        model_name (str): Registered model name in MLflow Model Registry.
        stage (str): Model lifecycle stage to query (e.g. "staging").
    """
    client = MlflowClient()

    # Retrieve the latest version metadata for the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, (
        f"No model found in the '{stage}' stage for '{model_name}'"
    )

    try:
        # Construct the MLflow model URI and attempt to load the artifact
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Confirm the model object was returned
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
