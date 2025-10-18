# force_mlflow_s3.py
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

print('=== Forcing MLflow S3 Configuration ===')

# Set environment variables explicitly
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

print('✅ AWS environment configured')

# Configure MLflow for S3
try:
    # Method 1: Set S3 tracking URI
    mlflow.set_tracking_uri('s3://insurance-charges-model-2025/mlflow')
    print('✅ MLflow S3 tracking URI configured')
    
    # Test by creating a run
    mlflow.set_experiment('s3-test-final')
    with mlflow.start_run(run_name='s3_direct_test'):
        mlflow.log_param('direct_s3', 'enabled')
        mlflow.log_metric('test_score', 0.99)
        print('✅ Direct S3 test successful!')
        
except Exception as e:
    print(f'❌ Direct S3 failed: {e}')
    
    # Fallback: Use our custom config
    print('Trying custom MLflow config...')
    from src.insurance_charges.utils.mlflow_config import MLflowConfig
    config = MLflowConfig()
    print(f'S3 Enabled: {config.s3_enabled}')
    info = config.get_storage_info()
    for key, value in info.items():
        print(f'{key}: {value}')