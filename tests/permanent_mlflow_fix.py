# permanent_mlflow_s3_fix.py
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

load_dotenv()

print("=== PERMANENT MLFLOW S3 FIX ===")

# Set environment variables PERMANENTLY for this session
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')

if aws_key and aws_secret:
    os.environ['AWS_ACCESS_KEY_ID'] = aws_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    print("✅ AWS credentials set in environment")
else:
    print("❌ AWS credentials missing in .env file")
    sys.exit(1)

# Now import and reconfigure MLflow
from src.insurance_charges.utils.mlflow_config import MLflowConfig

print("\n=== Reinitializing MLflow with S3 ===")
# Force reinitialization by creating new instance
mlflow_config = MLflowConfig()

# Get updated info
info = mlflow_config.get_storage_info()
print(f"✅ S3 Enabled: {info['s3_enabled']}")
print(f"✅ S3 Bucket: {info['s3_bucket']}")
print(f"✅ Local Enabled: {info['local_enabled']}")

if info['s3_enabled']:
    print("\n🎉 MLflow S3 is now PERMANENTLY enabled!")
    print("📦 Artifacts will be stored in S3 for all future runs")
else:
    print("\n❌ MLflow S3 still disabled - checking why...")
    
    # Debug S3 connection
    try:
        import boto3
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        print(f"✅ S3 connection successful. Buckets: {[b['Name'] for b in response['Buckets']]}")
        
        # Check specific bucket access
        try:
            s3.head_bucket(Bucket='insurance-charges-model-2025')
            print("✅ S3 bucket access confirmed")
        except Exception as e:
            print(f"❌ Cannot access bucket: {e}")
            
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")

print("\n=== Testing MLflow S3 with New Run ===")
try:
    import mlflow
    
    # Use the configured MLflow
    mlflow.set_experiment('permanent-s3-test')
    
    with mlflow.start_run(run_name='permanent_s3_verification'):
        mlflow.log_param('s3_permanent', 'enabled')
        mlflow.log_metric('verification_score', 0.99)
        
        # Create test artifact
        with open('test_s3_artifact.txt', 'w') as f:
            f.write('This should go to S3 via MLflow')
        
        mlflow.log_artifact('test_s3_artifact.txt')
        os.remove('test_s3_artifact.txt')
        
        print("✅ Test run completed with S3 artifacts!")
        
except Exception as e:
    print(f"❌ MLflow test failed: {e}")

print("\n=== Manual S3 Sync Check ===")
# Check if manual sync is needed
try:
    result = mlflow_config.sync_artifacts_to_s3()
    print(f"✅ Manual sync completed: {result}")
except Exception as e:
    print(f"❌ Manual sync failed: {e}")

print("\n🚀 MLflow S3 configuration updated! Future runs will use S3 artifacts.")