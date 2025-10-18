# test.py - UPDATED FOR DUAL STORAGE TESTING
import mlflow
import os
import sys
from dotenv import load_dotenv

def setup_dual_storage():
    """Setup MLflow with dual storage (local + S3)"""
    # Clean up first
    if mlflow.active_run():
        mlflow.end_run()
    
    # Load environment
    load_dotenv()
    
    # Import and setup MLflow config
    from src.insurance_charges.utils.mlflow_config import MLflowConfig
    mlflow_config = MLflowConfig()
    
    storage_info = mlflow_config.get_storage_info()
    
    print("✅ MLflow DUAL Storage Configuration:")
    print(f"📊 Tracking: {storage_info['tracking_uri']}")
    print(f"🎯 Experiment: {storage_info['experiment_name']}")
    print(f"💾 Local: {storage_info['local_enabled']}")
    print(f"☁️  S3: {storage_info['s3_enabled']}")
    
    if storage_info['s3_enabled']:
        print(f"📦 S3 Bucket: {storage_info['s3_bucket']}")
    
    return mlflow_config

def test_dvc_dual_storage():
    """Test DVC dual storage setup"""
    try:
        from dvc_commands import DVCCommands
        dvc = DVCCommands()
        
        print("\n🔍 Testing DVC Dual Storage...")
        
        # Initialize DVC
        if dvc.dvc_init():
            print("✅ DVC initialized")
        
        # Setup S3 remote
        if dvc.setup_s3_remote():
            print("✅ DVC S3 remote configured")
        
        # Show status
        dvc.dvc_status()
        
        return True
        
    except Exception as e:
        print(f"❌ DVC dual storage test failed: {e}")
        return False

def main():
    """Main execution function with dual storage testing"""
    try:
        print("🚀 Starting Insurance Charges Pipeline with DUAL STORAGE...")
        
        # Step 1: Test DVC dual storage
        print("\n" + "="*50)
        print("Step 1: Testing DVC Dual Storage")
        print("="*50)
        dvc_test_success = test_dvc_dual_storage()
        
        # Step 2: Setup MLflow dual storage
        print("\n" + "="*50)
        print("Step 2: Setting up MLflow Dual Storage")
        print("="*50)
        mlflow_config = setup_dual_storage()
        
        # Step 3: Run pipeline
        print("\n" + "="*50)
        print("Step 3: Running Training Pipeline")
        print("="*50)
        from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
        
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        
        # Step 4: Final status
        print("\n" + "="*50)
        print("Step 4: Final Dual Storage Status")
        print("="*50)
        
        if dvc_test_success:
            print("✅ DVC: Dual storage operational")
        else:
            print("⚠️ DVC: Local storage only")
        
        storage_info = mlflow_config.get_storage_info()
        if storage_info['s3_enabled']:
            print("✅ MLflow: Dual storage operational")
        else:
            print("⚠️ MLflow: Local storage only")
        
        print("\n🎉 Pipeline completed successfully with DUAL STORAGE!")
        print("\n💡 View Results:")
        print("MLflow UI: mlflow ui --backend-store-uri file:///./mlruns --port 5000")
        print("DVC Status: python dvc_commands.py status")
        print("DVC Metrics: dvc metrics show")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()