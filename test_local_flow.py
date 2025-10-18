# test_local_mlflow.py
import mlflow
import os

def test_local_mlflow():
    print("🧪 Testing LOCAL MLflow Storage...")
    
    # Force local storage
    mlflow.set_tracking_uri('file:///./mlruns')
    mlflow.set_experiment("test-local-storage")
    
    with mlflow.start_run(run_name="local_test"):
        # Log some test data
        mlflow.log_param("test_param", "hello")
        mlflow.log_metric("test_accuracy", 0.95)
        
        # Create a test artifact
        with open("test_artifact.txt", "w") as f:
            f.write("Test artifact content")
        
        mlflow.log_artifact("test_artifact.txt")
        os.remove("test_artifact.txt")  # Clean up
        
        print("✅ Test artifacts logged to local storage")
    
    print("🎯 Check if ./mlruns directory was created:")
    if os.path.exists("./mlruns"):
        print("✅ ./mlruns directory exists!")
        # List contents
        for root, dirs, files in os.walk("./mlruns"):
            for file in files:
                print(f"   📄 {os.path.join(root, file)}")
    else:
        print("❌ ./mlruns directory not created!")

if __name__ == "__main__":
    test_local_mlflow()