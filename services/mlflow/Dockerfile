# The upstream image lacks the dependencies of MLflow's basic-auth app
# (`pip install mlflow[auth]`). Pure-Python wheels, so it builds on any arch.
FROM ghcr.io/mlflow/mlflow:v3.17.0
RUN pip install --no-cache-dir "mlflow[auth]==3.17.0"
