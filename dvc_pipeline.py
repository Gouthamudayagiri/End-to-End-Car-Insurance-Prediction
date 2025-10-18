# dvc_pipeline.py - UPDATED WITH DUAL STORAGE SUPPORT
import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime

# Add src to path
sys.path.append('src')

# Import required libraries
import pandas as pd
import numpy as np

from insurance_charges.pipeline.training_pipeline import TrainPipeline
from insurance_charges.utils.main_utils import read_yaml_file, write_yaml_file

class DVCManager:
    """Manage DVC configuration for dual storage (local + S3)"""
    
    def __init__(self):
        self.remote_name = "s3-storage"
        self.bucket_name = "insurance-charges-model-2025"
        
    def setup_dvc_dual_storage(self):
        """Setup DVC with both local and S3 storage"""
        try:
            # Check if DVC is initialized
            if not os.path.exists('.dvc'):
                print("❌ DVC not initialized. Run 'dvc init' first.")
                return False
            
            # Check if S3 remote is configured
            result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
            if self.remote_name not in result.stdout:
                print(f"📦 Setting up DVC S3 remote: {self.remote_name}")
                
                # Add S3 remote
                remote_url = f"s3://{self.bucket_name}/dvc-storage"
                setup_cmd = [
                    'dvc', 'remote', 'add', 
                    '-d', self.remote_name, 
                    remote_url
                ]
                
                result = subprocess.run(setup_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ Could not setup S3 remote: {result.stderr}")
                    return False
                
                print(f"✅ DVC S3 remote configured: {remote_url}")
            else:
                print("✅ DVC S3 remote already configured")
            
            # Configure S3 credentials for DVC
            self._configure_dvc_s3_credentials()
            
            return True
            
        except Exception as e:
            print(f"❌ DVC dual storage setup failed: {e}")
            return False
    
    def _configure_dvc_s3_credentials(self):
        """Configure AWS credentials for DVC S3"""
        try:
            aws_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if aws_key and aws_secret:
                # Set AWS credentials for DVC
                subprocess.run([
                    'dvc', 'remote', 'modify', 
                    self.remote_name, 
                    'access_key_id', aws_key
                ], check=True)
                
                subprocess.run([
                    'dvc', 'remote', 'modify', 
                    self.remote_name, 
                    'secret_access_key', aws_secret
                ], check=True)
                
                print("✅ DVC S3 credentials configured")
            else:
                print("⚠️ AWS credentials not found for DVC S3")
                
        except Exception as e:
            print(f"⚠️ DVC S3 credential configuration failed: {e}")
    
    def push_to_remote(self):
        """Push DVC data to S3 remote"""
        try:
            print("📤 Pushing DVC data to S3 remote...")
            result = subprocess.run(['dvc', 'push', '-r', self.remote_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ DVC data pushed to S3 successfully")
                return True
            else:
                print(f"❌ DVC push failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ DVC push failed: {e}")
            return False
    
    def pull_from_remote(self):
        """Pull DVC data from S3 remote"""
        try:
            print("📥 Pulling DVC data from S3 remote...")
            result = subprocess.run(['dvc', 'pull', '-r', self.remote_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ DVC data pulled from S3 successfully")
                return True
            else:
                print(f"❌ DVC pull failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ DVC pull failed: {e}")
            return False
    
    def get_dvc_status(self):
        """Get DVC status including remote information"""
        try:
            # Get DVC status
            status_result = subprocess.run(['dvc', 'status'], capture_output=True, text=True)
            print("DVC Status:")
            print(status_result.stdout)
            
            # Get remote info
            remote_result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
            print("DVC Remotes:")
            print(remote_result.stdout)
            
            return True
            
        except Exception as e:
            print(f"❌ DVC status check failed: {e}")
            return False

class DVCPipeline:
    def __init__(self):
        self.pipeline = TrainPipeline()
        self.dvc_manager = DVCManager()
        self.data_ingestion_artifact = None
        self.data_validation_artifact = None
        self.data_transformation_artifact = None
        self.model_trainer_artifact = None
        
        # Setup DVC dual storage
        self.dvc_manager.setup_dvc_dual_storage()
        
    def run_data_ingestion(self):
        """Run data ingestion and save outputs for DVC"""
        print("📥 Running Data Ingestion...")
        self.data_ingestion_artifact = self.pipeline.start_data_ingestion()
        
        # Copy files to DVC-tracked locations
        os.makedirs('data/processed', exist_ok=True)
        shutil.copy(self.data_ingestion_artifact.trained_file_path, 'data/processed/train.csv')
        shutil.copy(self.data_ingestion_artifact.test_file_path, 'data/processed/test.csv')
        
        # Save metrics
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        metrics = {
            "data_ingestion": {
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "train_columns": list(train_df.columns),
                "test_columns": list(test_df.columns),
                "timestamp": datetime.now().isoformat(),
                "storage": "dual_local_s3"
            }
        }
        
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/data_ingestion.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print("✅ Data Ingestion completed")
        return self.data_ingestion_artifact
    
    def run_data_validation(self):
        """Run data validation and save outputs for DVC"""
        if self.data_ingestion_artifact is None:
            self.data_ingestion_artifact = self.pipeline.start_data_ingestion()
            
        print("🔍 Running Data Validation...")
        self.data_validation_artifact = self.pipeline.start_data_validation(self.data_ingestion_artifact)
        
        # Copy validation report
        os.makedirs('reports', exist_ok=True)
        if os.path.exists(self.data_validation_artifact.drift_report_file_path):
            shutil.copy(self.data_validation_artifact.drift_report_file_path, 'reports/drift_report.yaml')
        
        # Save metrics
        metrics = {
            "data_validation": {
                "validation_status": self.data_validation_artifact.validation_status,
                "message": self.data_validation_artifact.message,
                "drift_detected": "drift_report.yaml" in os.listdir('reports'),
                "timestamp": datetime.now().isoformat(),
                "storage": "dual_local_s3"
            }
        }
        
        with open('metrics/data_validation.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print("✅ Data Validation completed")
        return self.data_validation_artifact
    
    def run_data_transformation(self):
        """Run data transformation and save outputs for DVC"""
        if self.data_ingestion_artifact is None:
            self.data_ingestion_artifact = self.pipeline.start_data_ingestion()
        if self.data_validation_artifact is None:
            self.data_validation_artifact = self.pipeline.start_data_validation(self.data_ingestion_artifact)
            
        print("🔄 Running Data Transformation...")
        self.data_transformation_artifact = self.pipeline.start_data_transformation(
            self.data_ingestion_artifact, 
            self.data_validation_artifact
        )
        
        # Copy files to DVC-tracked locations
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/transformed', exist_ok=True)
        
        shutil.copy(self.data_transformation_artifact.transformed_object_file_path, 'models/preprocessor.pkl')
        shutil.copy(self.data_transformation_artifact.transformed_train_file_path, 'data/transformed/train.npy')
        shutil.copy(self.data_transformation_artifact.transformed_test_file_path, 'data/transformed/test.npy')
        
        # Save transformation info
        train_data = np.load('data/transformed/train.npy', allow_pickle=True)
        test_data = np.load('data/transformed/test.npy', allow_pickle=True)
        
        metrics = {
            "data_transformation": {
                "train_shape": train_data.shape,
                "test_shape": test_data.shape,
                "features_count": train_data.shape[1] - 1,  # excluding target
                "preprocessor_saved": "preprocessor.pkl" in os.listdir('models'),
                "timestamp": datetime.now().isoformat(),
                "storage": "dual_local_s3"
            }
        }
        
        with open('metrics/data_transformation.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print("✅ Data Transformation completed")
        return self.data_transformation_artifact
    
    def run_model_training(self):
        """Run model training and save outputs for DVC"""
        if self.data_transformation_artifact is None:
            if self.data_ingestion_artifact is None:
                self.data_ingestion_artifact = self.pipeline.start_data_ingestion()
            if self.data_validation_artifact is None:
                self.data_validation_artifact = self.pipeline.start_data_validation(self.data_ingestion_artifact)
            self.data_transformation_artifact = self.pipeline.start_data_transformation(
                self.data_ingestion_artifact, 
                self.data_validation_artifact
            )
            
        print("🤖 Running Model Training...")
        self.model_trainer_artifact = self.pipeline.start_model_trainer(self.data_transformation_artifact)
        
        # Copy model and reports
        shutil.copy(self.model_trainer_artifact.trained_model_file_path, 'models/insurance_model.pkl')
        
        # Save model report
        model_report_path = os.path.join(
            os.path.dirname(self.model_trainer_artifact.trained_model_file_path),
            'artifacts',
            'model_report.yaml'
        )
        if os.path.exists(model_report_path):
            shutil.copy(model_report_path, 'reports/model_report.yaml')
        
        # Save metrics
        metrics = {
            "model_training": {
                "r2_score": float(self.model_trainer_artifact.metric_artifact.r2_score),
                "rmse": float(self.model_trainer_artifact.metric_artifact.rmse),
                "mae": float(self.model_trainer_artifact.metric_artifact.mae),
                "best_model": self.model_trainer_artifact.model_name,
                "feature_count": self.model_trainer_artifact.feature_count,
                "model_saved": "insurance_model.pkl" in os.listdir('models'),
                "timestamp": datetime.now().isoformat(),
                "storage": "dual_local_s3"
            }
        }
        
        with open('metrics/model_training.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print("✅ Model Training completed")
        return self.model_trainer_artifact
    
    def run_full_pipeline(self):
        """Run complete pipeline with DVC dual storage outputs"""
        try:
            # Create directories
            os.makedirs('data/processed', exist_ok=True)
            os.makedirs('data/transformed', exist_ok=True)
            os.makedirs('models', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            os.makedirs('metrics', exist_ok=True)
            
            print("🚀 Starting Full DVC Pipeline with DUAL Storage...")
            
            # Run all pipeline steps
            self.run_data_ingestion()
            self.run_data_validation()
            self.run_data_transformation()
            self.run_model_training()
            
            # Save final pipeline metrics
            final_metrics = {
                "pipeline_execution": {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "model_performance": {
                        "r2_score": float(self.model_trainer_artifact.metric_artifact.r2_score),
                        "best_model": self.model_trainer_artifact.model_name,
                    },
                    "data_metrics": {
                        "train_samples": len(pd.read_csv('data/processed/train.csv')),
                        "test_samples": len(pd.read_csv('data/processed/test.csv'))
                    },
                    "artifacts_generated": {
                        "data_files": len(os.listdir('data/processed')) + len(os.listdir('data/transformed')),
                        "models": len(os.listdir('models')),
                        "reports": len(os.listdir('reports')),
                        "metrics": len(os.listdir('metrics'))
                    },
                    "storage_config": "dual_local_s3"
                }
            }
            
            with open('metrics/pipeline_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            # Push to S3 remote
            print("📤 Pushing artifacts to DVC S3 remote...")
            push_success = self.dvc_manager.push_to_remote()
            
            if push_success:
                final_metrics["pipeline_execution"]["s3_sync"] = "success"
            else:
                final_metrics["pipeline_execution"]["s3_sync"] = "failed"
            
            # Update metrics with sync status
            with open('metrics/pipeline_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2)
                
            print("🎉 Full DVC Pipeline completed successfully with DUAL storage!")
            
        except Exception as e:
            # Save error metrics
            error_metrics = {
                "pipeline_execution": {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "storage_config": "dual_local_s3"
                }
            }
            
            with open('metrics/pipeline_metrics.json', 'w') as f:
                json.dump(error_metrics, f, indent=2)
                
            print(f"❌ Pipeline failed: {e}")
            raise e

    def show_metrics(self):
        """Display DVC metrics"""
        try:
            print("📊 DVC Pipeline Metrics:")
            
            # Show DVC metrics
            subprocess.run(['dvc', 'metrics', 'show'], check=True)
            
            # Show storage info
            print("\n💾 Storage Configuration:")
            print("✅ Local: ./ (DVC tracked)")
            print("✅ S3: s3://insurance-charges-model-2025/dvc-storage")
            
        except Exception as e:
            print(f"❌ Could not display metrics: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run DVC pipeline with dual storage')
    parser.add_argument('--stage', type=str, choices=[
        'data_ingestion', 'data_validation', 'data_transformation', 
        'model_training', 'full', 'metrics', 'push', 'pull', 'status'
    ], default='full', help='Pipeline stage to run')
    
    args = parser.parse_args()
    pipeline = DVCPipeline()
    
    if args.stage == 'data_ingestion':
        pipeline.run_data_ingestion()
    elif args.stage == 'data_validation':
        pipeline.run_data_validation()
    elif args.stage == 'data_transformation':
        pipeline.run_data_transformation()
    elif args.stage == 'model_training':
        pipeline.run_model_training()
    elif args.stage == 'metrics':
        pipeline.show_metrics()
    elif args.stage == 'push':
        pipeline.dvc_manager.push_to_remote()
    elif args.stage == 'pull':
        pipeline.dvc_manager.pull_from_remote()
    elif args.stage == 'status':
        pipeline.dvc_manager.get_dvc_status()
    else:  # full
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()