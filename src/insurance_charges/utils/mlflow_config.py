import os
import mlflow
from src.insurance_charges.logger import logging

class MLflowConfig:
    def __init__(self, tracking_uri=None, experiment_name=None):
        # Use AWS if credentials available, otherwise local
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if aws_access_key and aws_secret_key:
            # AWS S3 tracking
            self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            self.artifact_location = f"s3://{os.getenv('MODEL_BUCKET_NAME', 'insurance-charges-model')}/mlflow-artifacts"
        else:
            # Local tracking
            self.tracking_uri = tracking_uri or 'file:///./mlruns'
            self.artifact_location = None
            
        self.experiment_name = experiment_name or "insurance-charges-production"
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configure MLflow with AWS S3 support"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment with S3 artifact location if available
            if self.artifact_location:
                mlflow.set_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
            else:
                mlflow.set_experiment(self.experiment_name)
            
            logging.info(f"✅ MLflow configured: {self.experiment_name}")
            logging.info(f"📊 Tracking URI: {self.tracking_uri}")
            if self.artifact_location:
                logging.info(f"📦 Artifact Location: {self.artifact_location}")
            
        except Exception as e:
            logging.error(f"❌ MLflow configuration failed: {e}")