# test_mlflow_fix.py
import os
from dotenv import load_dotenv

load_dotenv()

print('=== Debugging MLflow S3 ===')
print('Environment Variables:')

aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

if aws_key:
    print(f'AWS_ACCESS_KEY_ID: {aws_key[:10]}...')
else:
    print('❌ AWS_ACCESS_KEY_ID NOT FOUND')

if aws_secret:
    print(f'AWS_SECRET_ACCESS_KEY: {aws_secret[:10]}...')
else:
    print('❌ AWS_SECRET_ACCESS_KEY NOT FOUND')

print(f'AWS_DEFAULT_REGION: {os.getenv("AWS_DEFAULT_REGION")}')

# Test if MLflow can access S3 directly
try:
    import mlflow
    print(f'MLflow Tracking URI: {mlflow.get_tracking_uri()}')
    
    # Try to set S3 tracking URI
    mlflow.set_tracking_uri('s3://insurance-charges-model-2025/mlflow')
    print('✅ MLflow S3 tracking URI set')
except Exception as e:
    print(f'❌ MLflow S3 setup failed: {e}')