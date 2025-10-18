# test_minimal.py - Test pipeline without MLflow
import os
import sys

# Disable MLflow for this test
os.environ['MLFLOW_DISABLED'] = 'True'

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.insurance_charges.pipeline.training_pipeline import TrainPipeline

def run_minimal_test():
    try:
        print("🚀 Starting minimal pipeline test...")
        
        pipeline = TrainPipeline()
        
        # Test just data ingestion first
        print("📥 Testing data ingestion...")
        data_ingestion_artifact = pipeline.start_data_ingestion()
        print(f"✅ Data ingestion successful: {data_ingestion_artifact}")
        
        print("🎉 Minimal test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False

if __name__ == "__main__":
    run_minimal_test()