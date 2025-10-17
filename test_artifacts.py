# test_artifacts.py - Test MLflow artifact logging
import os
import sys
import mlflow
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_mlflow_artifacts():
    """Test if MLflow can log artifacts properly"""
    print("🧪 Testing MLflow Artifact Logging...")
    
    # Clean up first
    if mlflow.active_run():
        mlflow.end_run()
    
    try:
        with mlflow.start_run(run_name="artifact_test") as run:
            print(f"📊 Run started: {run.info.run_id}")
            
            # Test 1: Log metrics and params
            mlflow.log_param("test_param", "hello")
            mlflow.log_metric("test_accuracy", 0.95)
            print("✅ Parameters and metrics logged")
            
            # Test 2: Log a simple model
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            model = RandomForestRegressor(n_estimators=10)
            model.fit(X, y)
            
            mlflow.sklearn.log_model(model, "test_model")
            print("✅ Model artifact logged")
            
            # Test 3: Log a file artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test artifact content")
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "test_artifacts")
            os.unlink(temp_path)  # Clean up
            print("✅ File artifact logged")
            
            # Test 4: Log multiple artifacts from a directory
            temp_dir = tempfile.mkdtemp()
            with open(os.path.join(temp_dir, "file1.txt"), 'w') as f:
                f.write("File 1 content")
            with open(os.path.join(temp_dir, "file2.txt"), 'w') as f:
                f.write("File 2 content")
            
            mlflow.log_artifacts(temp_dir, "multiple_artifacts")
            print("✅ Multiple artifacts logged")
            
            print(f"🎯 Artifact location: {run.info.artifact_uri}")
            
        print("\n✅ ALL ARTIFACT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Artifact test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mlflow_artifacts()
    if success:
        print("\n🎯 MLflow artifact logging is working! Update your pipeline code.")
    else:
        print("\n⚠️ There are issues with MLflow artifact logging.")