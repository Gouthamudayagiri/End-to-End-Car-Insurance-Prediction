# src/insurance_charges/utils/mlflow_config.py
import os
import mlflow
from src.insurance_charges.logger import logging

class MLflowConfig:
    def __init__(self, tracking_uri=None, experiment_name=None):
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'file:///./mlruns')
        self.experiment_name = experiment_name or "insurance-charges-production"
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configure MLflow with AWS S3 support"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment (creates if doesn't exist)
            mlflow.set_experiment(self.experiment_name)
            
            # Configure S3 artifact storage
            # MLflow automatically uses AWS credentials from environment
            
            logging.info(f"✅ MLflow configured: {self.experiment_name}")
            logging.info(f"📊 Tracking URI: {self.tracking_uri}")
            
        except Exception as e:
            logging.error(f"❌ MLflow configuration failed: {e}")
            # Don't raise exception - allow fallback to local tracking
    
    def get_experiment_id(self):
        """Get current experiment ID"""
        return mlflow.get_experiment_by_name(self.experiment_name).experiment_id