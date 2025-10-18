# test_comprehensive.py - Step-by-step pipeline test
import os
import sys
import mlflow

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
from src.insurance_charges.logger import logging

def cleanup_mlflow():
    """Ensure clean MLflow state"""
    try:
        if mlflow.active_run():
            mlflow.end_run()
            print("✅ Cleaned up active MLflow run")
    except Exception as e:
        print(f"⚠️ MLflow cleanup warning: {e}")

def test_pipeline_step_by_step():
    """Test pipeline step by step with MLflow"""
    try:
        print("=" * 80)
        print("🚀 STARTING COMPREHENSIVE PIPELINE TEST")
        print("=" * 80)
        
        # Clean up first
        cleanup_mlflow()
        
        # Initialize pipeline
        pipeline = TrainPipeline()
        
        # Step 1: Data Ingestion
        print("\n📥 STEP 1: Testing Data Ingestion...")
        data_ingestion_artifact = pipeline.start_data_ingestion()
        print(f"✅ Data Ingestion Successful: {data_ingestion_artifact}")
        
        # Step 2: Data Analysis
        print("\n📊 STEP 2: Testing Data Analysis...")
        analysis_report = pipeline.start_data_analysis(data_ingestion_artifact)
        print(f"✅ Data Analysis Successful: {analysis_report.get('quality_score', 'N/A')}")
        
        # Step 3: Data Validation
        print("\n🔍 STEP 3: Testing Data Validation...")
        data_validation_artifact = pipeline.start_data_validation(data_ingestion_artifact)
        print(f"✅ Data Validation Successful: {data_validation_artifact.validation_status}")
        
        # Step 4: Data Transformation
        print("\n🔄 STEP 4: Testing Data Transformation...")
        data_transformation_artifact = pipeline.start_data_transformation(
            data_ingestion_artifact, 
            data_validation_artifact
        )
        print(f"✅ Data Transformation Successful: {data_transformation_artifact}")
        
        # Step 5: Model Training (with MLflow)
        print("\n🤖 STEP 5: Testing Model Training with MLflow...")
        
        # Start MLflow run for this test
        mlflow_run = mlflow.start_run(run_name="comprehensive_test", experiment_id="0")
        print(f"📊 MLflow Run Started: {mlflow_run.info.run_id}")
        
        try:
            model_trainer_artifact = pipeline.start_model_trainer(data_transformation_artifact)
            print(f"✅ Model Training Successful: {model_trainer_artifact}")
            
            # Log test metrics
            mlflow.log_metrics({
                "test_r2_score": model_trainer_artifact.metric_artifact.r2_score,
                "test_rmse": model_trainer_artifact.metric_artifact.rmse,
                "test_mae": model_trainer_artifact.metric_artifact.mae
            })
            print("✅ Metrics logged to MLflow")
            
        finally:
            # Always end the MLflow run
            mlflow.end_run()
            print("✅ MLflow run ended")
        
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        # Ensure MLflow run is ended even on failure
        try:
            if mlflow.active_run():
                mlflow.end_run()
                print("✅ Cleaned up MLflow run after failure")
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_pipeline_step_by_step()
    if success:
        print("\n🎯 All tests passed! Your pipeline is ready for full execution.")
        print("\nNext step: Run 'python test.py' for the full pipeline")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")