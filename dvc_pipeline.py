# Add this at the VERY TOP of dvc_pipeline.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# dvc_pipeline.py - UPDATED WITH STAGE SUPPORT
import os
import sys
import json
import shutil
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

# Import required libraries
import pandas as pd
import numpy as np

from insurance_charges.pipeline.training_pipeline import TrainPipeline
from insurance_charges.utils.main_utils import read_yaml_file, write_yaml_file

class DVCPipeline:
    def __init__(self):
        self.pipeline = TrainPipeline()
        self.data_ingestion_artifact = None
        self.data_validation_artifact = None
        self.data_transformation_artifact = None
        self.model_trainer_artifact = None
        
    def run_data_ingestion(self):
        """Run data ingestion and save outputs for DVC"""
        print("📥 Running Data Ingestion...")
        self.data_ingestion_artifact = self.pipeline.start_data_ingestion()
        
        # Copy files to DVC-tracked locations
        os.makedirs('data/processed', exist_ok=True)
        shutil.copy(self.data_ingestion_artifact.trained_file_path, 'data/processed/train.csv')
        shutil.copy(self.data_ingestion_artifact.test_file_path, 'data/processed/test.csv')
        
        # Save metrics
        metrics = {
            "data_ingestion": {
                "train_samples": len(pd.read_csv('data/processed/train.csv')),
                "test_samples": len(pd.read_csv('data/processed/test.csv')),
                "timestamp": datetime.now().isoformat()
            }
        }
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
        if os.path.exists(self.data_validation_artifact.drift_report_file_path):
            shutil.copy(self.data_validation_artifact.drift_report_file_path, 'reports/drift_report.yaml')
        
        # Save metrics
        metrics = {
            "data_validation": {
                "validation_status": self.data_validation_artifact.validation_status,
                "message": self.data_validation_artifact.message,
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
            }
        }
        with open('metrics/model_training.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print("✅ Model Training completed")
        return self.model_trainer_artifact
    
    def run_full_pipeline(self):
        """Run complete pipeline with DVC-compatible outputs"""
        try:
            # Create directories
            os.makedirs('data/processed', exist_ok=True)
            os.makedirs('data/transformed', exist_ok=True)
            os.makedirs('models', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            os.makedirs('metrics', exist_ok=True)
            
            print("🚀 Starting Full DVC Pipeline...")
            
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
                    }
                }
            }
            with open('metrics/pipeline_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2)
                
            print("🎉 Full DVC Pipeline completed successfully!")
            
        except Exception as e:
            # Save error metrics
            error_metrics = {
                "pipeline_execution": {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
            with open('metrics/pipeline_metrics.json', 'w') as f:
                json.dump(error_metrics, f, indent=2)
            print(f"❌ Pipeline failed: {e}")
            raise e

def main():
    parser = argparse.ArgumentParser(description='Run DVC pipeline stages')
    parser.add_argument('--stage', type=str, choices=[
        'data_ingestion', 'data_validation', 'data_transformation', 
        'model_training', 'full'
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
    else:  # full
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()