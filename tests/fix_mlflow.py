# fix_mlflow.py - Run this before your pipeline to clean up MLflow state
import mlflow
import os

def cleanup_mlflow():
    """Clean up any active MLflow runs"""
    try:
        # End any active runs
        if mlflow.active_run():
            print("Ending active MLflow run...")
            mlflow.end_run()
            print("✅ Active run ended")
        else:
            print("✅ No active MLflow runs")
            
        # Set tracking URI
        mlflow.set_tracking_uri('file:///./mlruns')
        print("✅ MLflow tracking URI set")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    cleanup_mlflow()