# test_simple.py - Test without MLflow first
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test if basic imports work"""
    try:
        from src.insurance_charges.exception import InsuranceException
        from src.insurance_charges.logger import logging
        print("✅ Basic imports work")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_mlflow_import():
    """Test if MLflow imports work"""
    try:
        import mlflow
        print(f"✅ MLflow version: {mlflow.__version__}")
        return True
    except Exception as e:
        print(f"❌ MLflow import failed: {e}")
        return False

def test_pipeline_creation():
    """Test if pipeline can be created"""
    try:
        from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
        pipeline = TrainPipeline()
        print("✅ Pipeline creation successful")
        return True
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running basic tests...")
    
    tests_passed = 0
    tests_total = 3
    
    if test_basic_imports():
        tests_passed += 1
    
    if test_mlflow_import():
        tests_passed += 1
        
    if test_pipeline_creation():
        tests_passed += 1
    
    print(f"\n🎯 Test Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("🚀 All tests passed! You can now run the pipeline.")
    else:
        print("⚠️ Some tests failed. Please fix the issues above.")