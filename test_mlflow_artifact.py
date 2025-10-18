# test_mlflow_s3.py
import os
import mlflow
from dotenv import load_dotenv

def test_mlflow_s3_artifacts():
    """Test if MLflow can store artifacts in S3"""
    load_dotenv()
    
    print("🧪 Testing MLflow S3 Artifact Storage...")
    
    # Set up MLflow
    mlflow.set_tracking_uri('file:///./mlruns')
    mlflow.set_experiment("s3-test-experiment")
    
    # Start a run
    with mlflow.start_run(run_name="s3_artifact_test"):
        print("✅ MLflow run started")
        
        # Log some parameters and metrics
        mlflow.log_param("test_param", "s3_artifact_test")
        mlflow.log_metric("test_accuracy", 0.95)
        
        # Create a test artifact
        test_content = "This is a test artifact for S3 storage"
        with open("test_artifact.txt", "w") as f:
            f.write(test_content)
        
        try:
            # Log artifact - this should go to S3 if configured properly
            mlflow.log_artifact("test_artifact.txt", "s3_test_artifacts")
            print("✅ Artifact logged successfully")
            
            # Check if artifact was stored in S3
            import boto3
            s3 = boto3.client('s3',
                             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
            
            # Try to list S3 artifacts
            bucket_name = "insurance-charges-model-2025"
            try:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix="mlflow/", Delimiter="/")
                if 'Contents' in response:
                    print("✅ S3 artifacts found:")
                    for obj in response['Contents'][:5]:  # Show first 5
                        print(f"   📦 {obj['Key']} ({obj['Size']} bytes)")
                else:
                    print("ℹ️  No artifacts found in S3 yet")
                    
            except Exception as e:
                print(f"⚠️ Could not list S3 artifacts: {e}")
            
        except Exception as e:
            print(f"❌ Failed to log artifact to S3: {e}")
            print("💡 Check MLflow S3 configuration")
        
        # Clean up
        if os.path.exists("test_artifact.txt"):
            os.remove("test_artifact.txt")
    
    print("🎯 S3 artifact test completed")

if __name__ == "__main__":
    test_mlflow_s3_artifacts()