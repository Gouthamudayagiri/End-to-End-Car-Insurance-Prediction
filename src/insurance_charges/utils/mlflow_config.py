# src/insurance_charges/utils/mlflow_config.py
import os
import mlflow
import boto3
from botocore.exceptions import ClientError
from src.insurance_charges.logger import logging

class MLflowConfig:
    def __init__(self, tracking_uri=None, experiment_name=None):
        """
        MLflow configuration with DUAL STORAGE (Local + S3)
        """
        try:
            self.experiment_name = experiment_name or "insurance-charges-production"
            self.bucket_name = "insurance-charges-model-2025"
            
            # Configure AWS credentials
            self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            # Initialize storage flags
            self.s3_enabled = False
            self.local_enabled = True  # Always enable local
            
            # Setup storage backends
            self._setup_dual_storage()
            
            logging.info(f"✅ MLflow configured with DUAL storage")
            logging.info(f"📍 Local: ./mlruns")
            if self.s3_enabled:
                logging.info(f"📍 S3: s3://{self.bucket_name}/mlflow")
            
        except Exception as e:
            logging.error(f"❌ MLflow configuration failed: {e}")
            # Fallback to local only
            self._setup_local_only()
    
    def _setup_dual_storage(self):
        """Setup both local and S3 storage backends"""
        try:
            # Step 1: Configure AWS for S3
            self._configure_aws_environment()
            
            # Step 2: Setup local tracking (ALWAYS AVAILABLE)
            self._setup_local_tracking()
            
            # Step 3: Setup experiment with dual artifact locations
            self._setup_experiment_with_dual_storage()
            
        except Exception as e:
            logging.warning(f"⚠️ Dual storage setup failed, falling back to local: {e}")
            self._setup_local_only()
    
    def _configure_aws_environment(self):
        """Configure AWS environment for S3 access"""
        try:
            if self.aws_access_key and self.aws_secret_key:
                # Set AWS environment variables
                os.environ['AWS_ACCESS_KEY_ID'] = self.aws_access_key
                os.environ['AWS_SECRET_ACCESS_KEY'] = self.aws_secret_key
                os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
                
                # Test S3 connection
                if self._test_s3_connection():
                    self.s3_enabled = True
                    logging.info("✅ AWS S3 connection validated")
                else:
                    logging.warning("⚠️ S3 connection test failed")
                    self.s3_enabled = False
            else:
                logging.info("💡 S3 disabled: No AWS credentials found")
                self.s3_enabled = False
                
        except Exception as e:
            logging.warning(f"⚠️ AWS configuration failed: {e}")
            self.s3_enabled = False
    
    def _test_s3_connection(self):
        """Test S3 connectivity and bucket access"""
        try:
            s3_client = boto3.client('s3')
            
            # Test basic S3 access
            s3_client.list_buckets()
            
            # Test bucket access
            try:
                s3_client.head_bucket(Bucket=self.bucket_name)
                logging.info(f"✅ S3 bucket accessible: {self.bucket_name}")
                return True
            except ClientError:
                logging.warning(f"⚠️ S3 bucket doesn't exist: {self.bucket_name}")
                # Try to create bucket
                try:
                    s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': 'us-east-1'}
                    )
                    logging.info(f"✅ Created S3 bucket: {self.bucket_name}")
                    return True
                except ClientError as e:
                    logging.warning(f"⚠️ Could not create S3 bucket: {e}")
                    return False
                    
        except Exception as e:
            logging.warning(f"⚠️ S3 connection test failed: {e}")
            return False
    
    def _setup_local_tracking(self):
        """Setup local MLflow tracking"""
        try:
            # Always use local tracking store
            local_tracking_uri = 'file:///./mlruns'
            mlflow.set_tracking_uri(local_tracking_uri)
            self.local_tracking_uri = local_tracking_uri
            logging.info(f"✅ Local tracking: {local_tracking_uri}")
        except Exception as e:
            logging.error(f"❌ Local tracking setup failed: {e}")
            raise e
    
    def _setup_experiment_with_dual_storage(self):
        """Setup experiment with both local and S3 artifact locations"""
        try:
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                # Create new experiment
                mlflow.create_experiment(self.experiment_name)
                logging.info(f"✅ Created experiment: {self.experiment_name}")
            else:
                # Use existing experiment
                mlflow.set_experiment(self.experiment_name)
                logging.info(f"✅ Using existing experiment: {self.experiment_name}")
            
            # Set the experiment
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            logging.info(f"🎯 Experiment ID: {self.experiment.experiment_id}")
            
        except Exception as e:
            logging.error(f"❌ Experiment setup failed: {e}")
            raise e
    
    def _setup_local_only(self):
        """Fallback to local-only configuration"""
        try:
            mlflow.set_tracking_uri('file:///./mlruns')
            mlflow.set_experiment(self.experiment_name)
            self.s3_enabled = False
            self.local_enabled = True
            logging.info("🔄 Fallback to local-only MLflow tracking")
        except Exception as e:
            logging.error(f"❌ Local-only setup failed: {e}")
    
    def get_artifact_locations(self):
        """Get available artifact locations"""
        locations = {
            "local": "./mlruns",
            "tracking_uri": self.local_tracking_uri,
            "experiment_name": self.experiment_name
        }
        
        if self.s3_enabled:
            locations["s3"] = f"s3://{self.bucket_name}/mlflow"
        
        return locations
    
    def sync_artifacts_to_s3(self, run_id=None):
        """Manually sync artifacts from local to S3"""
        if not self.s3_enabled:
            logging.warning("⚠️ S3 not enabled, skipping artifact sync")
            return False
        
        try:
            import subprocess
            import datetime
            
            local_mlruns = "./mlruns"
            s3_destination = f"s3://{self.bucket_name}/mlflow"
            
            if os.path.exists(local_mlruns):
                logging.info(f"📤 Syncing {local_mlruns} to {s3_destination}")
                
                sync_command = [
                    "aws", "s3", "sync", 
                    local_mlruns, 
                    s3_destination,
                    "--delete",
                    "--size-only"
                ]
                
                result = subprocess.run(sync_command, capture_output=True, text=True)
                
                if result.returncode == 0:
                    if result.stdout:
                        logging.info(f"✅ S3 sync completed: {result.stdout}")
                    else:
                        logging.info("✅ S3 sync completed (no changes)")
                    return True
                else:
                    logging.warning(f"⚠️ S3 sync failed: {result.stderr}")
                    return False
            else:
                logging.warning("⚠️ No local mlruns directory to sync")
                return False
                
        except Exception as e:
            logging.warning(f"⚠️ Manual S3 sync failed: {e}")
            return False
    
    def log_artifacts_dual(self, local_path, artifact_path=None, run_id=None):
        """Log artifacts to both local and S3 storage"""
        try:
            # Always log to local
            mlflow.log_artifacts(local_path, artifact_path)
            logging.info(f"✅ Artifacts logged locally: {local_path}")
            
            # Also sync to S3 if enabled
            if self.s3_enabled:
                self.sync_artifacts_to_s3(run_id)
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Dual artifact logging failed: {e}")
            return False
    
    def get_storage_info(self):
        """Get comprehensive storage information"""
        return {
            "tracking_uri": self.local_tracking_uri,
            "experiment_name": self.experiment_name,
            "local_enabled": self.local_enabled,
            "s3_enabled": self.s3_enabled,
            "s3_bucket": self.bucket_name if self.s3_enabled else None,
            "artifact_locations": self.get_artifact_locations()
        }