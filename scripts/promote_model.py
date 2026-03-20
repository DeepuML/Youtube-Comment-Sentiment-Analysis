import os
import mlflow

def promote_model():
    """Promote the latest Staging model version to Production in the MLflow Model Registry.

    Steps performed:
    1. Connect to the remote MLflow tracking server.
    2. Retrieve the latest model version currently in the 'Staging' stage.
    3. Archive every model version currently in 'Production'.
    4. Transition the Staging version to 'Production'.
    """
    # Connect to the remote MLflow tracking server
    mlflow.set_tracking_uri("http://34.224.212.114:8000/")

    client = mlflow.MlflowClient()

    model_name = "yt_chrome_plugin_model"
    # Retrieve the latest version available in the Staging stage
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive all existing Production versions before promoting the new one
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the Staging version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()