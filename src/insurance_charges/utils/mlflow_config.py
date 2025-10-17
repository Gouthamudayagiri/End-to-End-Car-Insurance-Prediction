# src/insurance_charges/utils/mlflow_config.py
import os
import mlflow
from src.insurance_charges.logger import logging

class MLflowConfig:
    def __init__(self, tracking_uri=None, experiment_name=None):
        # Use S3 for artifacts if AWS credentials available
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if self.aws_access_key and self.aws_secret_key:
            # S3 artifact location
            self.artifact_location = f"s3://insurance-charges-model-2025/mlflow-artifacts"
            self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'file:///./mlruns')
            logging.info("✅ MLflow configured with S3 artifact storage")
        else:
            # Local storage
            self.artifact_location = None
            self.tracking_uri = tracking_uri or 'file:///./mlruns'
            logging.info("✅ MLflow configured with local storage (no AWS credentials)")
            
        self.experiment_name = experiment_name or "insurance-charges-production"
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configure MLflow with S3 artifact storage"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment with S3 artifact location
            if self.artifact_location:
                try:
                    # Try to create experiment with S3 artifact location
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    if experiment is None:
                        mlflow.create_experiment(
                            self.experiment_name,
                            artifact_location=self.artifact_location
                        )
                        logging.info(f"✅ Created experiment with S3 artifacts: {self.experiment_name}")
                    else:
                        logging.info(f"✅ Using existing experiment: {self.experiment_name}")
                except Exception as e:
                    logging.warning(f"Could not configure S3 artifacts, using local: {e}")
                    mlflow.set_experiment(self.experiment_name)
            else:
                mlflow.set_experiment(self.experiment_name)
            
            logging.info(f"📊 MLflow Tracking URI: {self.tracking_uri}")
            if self.artifact_location:
                logging.info(f"📦 Artifact Location: {self.artifact_location}")
            
        except Exception as e:
            logging.error(f"❌ MLflow configuration failed: {e}")
    
    def get_experiment_id(self):
        """Get current experiment ID"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            return experiment.experiment_id if experiment else None
        except:
            return None