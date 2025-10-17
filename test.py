# test.py - Updated with MLflow cleanup
from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
import mlflow

def cleanup_mlflow():
    """Clean up any active MLflow runs before starting"""
    try:
        if mlflow.active_run():
            mlflow.end_run()
            print("✅ Cleaned up active MLflow run")
    except:
        pass

if __name__ == "__main__":
    # Clean up before starting
    cleanup_mlflow()
    
    # Run pipeline
    pipeline = TrainPipeline()
    pipeline.run_pipeline()