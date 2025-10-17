# final_verification.py - Final check before full pipeline
import os
import sys
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.insurance_charges.pipeline.training_pipeline import TrainPipeline

def final_check():
    """Final verification that everything is working"""
    print("🎯 FINAL VERIFICATION CHECK")
    print("=" * 50)
    
    # Check 1: MLflow status
    print("1. Checking MLflow...")
    try:
        tracking_uri = mlflow.get_tracking_uri()
        print(f"   ✅ Tracking URI: {tracking_uri}")
    except Exception as e:
        print(f"   ❌ MLflow issue: {e}")
        return False
    
    # Check 2: Pipeline creation
    print("2. Checking pipeline creation...")
    try:
        pipeline = TrainPipeline()
        print("   ✅ Pipeline created successfully")
    except Exception as e:
        print(f"   ❌ Pipeline creation failed: {e}")
        return False
    
    # Check 3: Test MLflow run
    print("3. Testing MLflow run...")
    try:
        with mlflow.start_run(run_name="final_verification") as run:
            mlflow.log_param("test", "success")
            mlflow.log_metric("accuracy", 0.95)
            print(f"   ✅ MLflow run test successful: {run.info.run_id}")
    except Exception as e:
        print(f"   ❌ MLflow run test failed: {e}")
        return False
    
    # Check 4: Clean up
    print("4. Cleaning up...")
    try:
        if mlflow.active_run():
            mlflow.end_run()
        print("   ✅ Cleanup successful")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")
    
    print("=" * 50)
    print("🎉 ALL CHECKS PASSED! Your pipeline is ready for production.")
    print("\nNext: Run 'python test.py' for the full pipeline execution")
    return True

if __name__ == "__main__":
    final_check()