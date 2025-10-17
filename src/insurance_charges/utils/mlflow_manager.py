import mlflow
from src.insurance_charges.constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, S3_ARTIFACT_LOCATION
from src.insurance_charges.logger import logging

class MLflowManager:
    @staticmethod
    def initialize():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logging.info("🎯 MLflow Manager initialized")