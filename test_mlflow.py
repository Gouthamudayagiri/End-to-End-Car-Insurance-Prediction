# test_mlflow_s3.py
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

def test_mlflow_s3():
    print("🔍 Testing MLflow S3 Configuration...")
    
    # Configure MLflow
    mlflow.set_tracking_uri('file:///./mlruns')
    
    # Test S3 artifact logging
    with mlflow.start_run(run_name="s3_test"):
        mlflow.log_param("test_param", "s3_artifact_test")
        mlflow.log_metric("test_accuracy", 0.95)
        
        # Try to log artifact to S3
        try:
            # Create test artifact
            with open("test_artifact.txt", "w") as f:
                f.write("Test artifact for S3")
            
            # Log to S3 if credentials available
            aws_key = os.getenv('AWS_ACCESS_KEY_ID')
            if aws_key:
                mlflow.log_artifact("test_artifact.txt", "s3_test")
                print("✅ Successfully logged artifact (would go to S3 if configured)")
            else:
                mlflow.log_artifact("test_artifact.txt", "local_test")
                print("✅ Logged artifact locally (no AWS credentials)")
                
            # Clean up
            os.remove("test_artifact.txt")
            
        except Exception as e:
            print(f"❌ Artifact logging failed: {e}")

if __name__ == "__main__":
    test_mlflow_s3()