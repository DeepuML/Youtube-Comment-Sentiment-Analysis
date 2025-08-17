import mlflow
import random

# Set the MLflow tracking URI
<<<<<<< HEAD
mlflow.set_tracking_uri("http://52.204.155.208:8000/")
=======
mlflow.set_tracking_uri("http://13.220.125.159:8000/")
>>>>>>> f5495bf (Added CI CD Workflow)

# Start an MLflow run
with mlflow.start_run():
    # Log some random parameters
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log some random metrics
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")