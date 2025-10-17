# test.py - Updated with MLflow cleanup
import mlflow
import os
import sys

def cleanup_mlflow():
    """Clean up any active MLflow runs before starting"""
    try:
        # End any active runs
        if mlflow.active_run():
            mlflow.end_run()
            print("✅ Cleaned up active MLflow run")
        
        # Clear MLflow environment variables that might cause conflicts
        mlflow_env_vars = ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID', 'MLFLOW_TRACKING_URI']
        for var in mlflow_env_vars:
            if var in os.environ:
                del os.environ[var]
                
    except Exception as e:
        print(f"⚠️ MLflow cleanup warning: {e}")

def main():
    """Main execution function"""
    try:
        print("🚀 Starting Insurance Charges Training Pipeline...")
        
        # Clean up before starting
        cleanup_mlflow()
        
        # Import and run pipeline
        from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
        
        # Run pipeline
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        
        print("🎉 Pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        # Ensure cleanup even on failure
        cleanup_mlflow()
        sys.exit(1)

if __name__ == "__main__":
    main()