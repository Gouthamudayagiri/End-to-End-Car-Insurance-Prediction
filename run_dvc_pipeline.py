# run_dvc_pipeline.py
import os
import subprocess
import sys
import mlflow
from datetime import datetime

def run_dvc_pipeline():
    """Run DVC pipeline with MLflow tracking"""
    
    # Add src to path
    sys.path.append('src')
    
    run_name = f"dvc_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        try:
            # Log DVC configuration
            mlflow.log_artifact("params.yaml", "dvc_config")
            mlflow.log_artifact("dvc.yaml", "dvc_pipeline")
            
            # Log current git commit (for reproducibility)
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, text=True
            )
            if git_commit.returncode == 0:
                mlflow.log_param("git_commit", git_commit.stdout.strip())
            
            # Run DVC pipeline
            print("🚀 Starting DVC pipeline...")
            result = subprocess.run(
                ["dvc", "repro"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print("✅ DVC pipeline completed successfully")
                
                # Log DVC state
                mlflow.log_artifact("dvc.lock", "dvc_state")
                
                # Log DVC metrics
                metrics_result = subprocess.run(
                    ["dvc", "metrics", "show"], 
                    capture_output=True, 
                    text=True
                )
                if metrics_result.returncode == 0:
                    mlflow.log_text(metrics_result.stdout, "dvc_metrics.txt")
                
                # Log DVC pipeline visualization
                dag_result = subprocess.run(
                    ["dvc", "dag"], 
                    capture_output=True, 
                    text=True
                )
                if dag_result.returncode == 0:
                    mlflow.log_text(dag_result.stdout, "dvc_pipeline_dag.txt")
                
                mlflow.log_param("dvc_status", "success")
                print("🎉 Pipeline completed with MLflow tracking!")
                
            else:
                print(f"❌ DVC pipeline failed: {result.stderr}")
                mlflow.log_param("dvc_status", "failed")
                mlflow.log_text(result.stderr, "dvc_error.log")
                raise Exception(f"DVC pipeline failed: {result.stderr}")
                
        except Exception as e:
            mlflow.log_param("dvc_status", "error")
            mlflow.log_param("error", str(e))
            raise e

if __name__ == "__main__":
    run_dvc_pipeline()