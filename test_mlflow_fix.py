# # # test_mlflow_fix.py
# # import os
# # import sys
# # sys.path.append('src')

# # from src.insurance_charges.utils.mlflow_config import MLflowConfig

# # def test_mlflow_config():
# #     print("🧪 Testing MLflow Configuration...")
    
# #     try:
# #         mlflow_config = MLflowConfig()
# #         storage_info = mlflow_config.get_storage_info()
        
# #         print("✅ MLflow Configuration Successful!")
# #         print(f"Storage Mode: {storage_info['storage_mode']}")
# #         print(f"Tracking URI: {storage_info['tracking_uri']}")
# #         print(f"Local Tracking URI: {storage_info.get('local_tracking_uri', 'N/A')}")
# #         print(f"Artifact Location: {storage_info.get('artifact_location', 'N/A')}")
# #         print(f"AWS Configured: {storage_info['aws_configured']}")
        
# #         # Check all required keys are present
# #         required_keys = ['storage_mode', 'tracking_uri', 'local_tracking_uri', 'artifact_location', 'aws_configured']
# #         missing_keys = [key for key in required_keys if key not in storage_info]
        
# #         if missing_keys:
# #             print(f"❌ Missing keys: {missing_keys}")
# #             return False
# #         else:
# #             print("✅ All required keys present!")
# #             return True
            
# #     except Exception as e:
# #         print(f"❌ MLflow configuration test failed: {e}")
# #         return False

# # if __name__ == "__main__":
# #     test_mlflow_config()

# # test_mlflow_complete.py
# import os
# import sys
# sys.path.append('src')

# from src.insurance_charges.utils.mlflow_config import MLflowConfig

# def test_all_mlflow_keys():
#     print("🧪 Testing ALL MLflow Configuration Keys...")
    
#     try:
#         mlflow_config = MLflowConfig()
#         storage_info = mlflow_config.get_storage_info()
        
#         print("✅ MLflow Configuration Successful!")
        
#         # Check all possible keys
#         all_keys = [
#             'storage_mode', 'tracking_uri', 'local_tracking_uri', 
#             's3_tracking_uri', 's3_artifact_location', 'artifact_location',
#             'experiment_name', 'aws_configured', 's3_accessible'
#         ]
        
#         print("\n📊 Storage Information:")
#         for key in all_keys:
#             value = storage_info.get(key, "❌ MISSING")
#             print(f"   {key}: {value}")
        
#         # Check if all required keys are present
#         required_keys = ['storage_mode', 'tracking_uri', 'local_tracking_uri', 'artifact_location', 'aws_configured']
#         missing_keys = [key for key in required_keys if key not in storage_info]
        
#         if missing_keys:
#             print(f"\n❌ Missing required keys: {missing_keys}")
#             return False
#         else:
#             print(f"\n✅ All {len(required_keys)} required keys present!")
#             return True
            
#     except Exception as e:
#         print(f"❌ MLflow configuration test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# if __name__ == "__main__":
#     success = test_all_mlflow_keys()
#     if success:
#         print("\n🎯 MLflow configuration is ready for pipeline!")
#     else:
#         print("\n⚠️ MLflow configuration needs fixes!")

# test_pipeline_s3.py
import os
import mlflow
from dotenv import load_dotenv

def test_pipeline_s3_artifacts():
    """Test if pipeline artifacts go to S3"""
    load_dotenv()
    
    print("🧪 Testing Pipeline S3 Artifact Storage...")
    
    # Import and run pipeline
    from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
    
    # Run a quick pipeline test
    pipeline = TrainPipeline()
    
    # Just test data ingestion and a small part to see if S3 works
    print("📥 Testing data ingestion...")
    data_ingestion_artifact = pipeline.start_data_ingestion()
    print(f"✅ Data ingestion: {data_ingestion_artifact}")
    
    # Check S3 after pipeline components
    check_s3_artifacts()
    
    print("🎯 Pipeline S3 test completed")

def check_s3_artifacts():
    """Check what's in S3"""
    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = "insurance-charges-model-2025"
    
    if not aws_key or not aws_secret:
        print("❌ AWS credentials not available")
        return
    
    try:
        s3 = boto3.client('s3',
                         aws_access_key_id=aws_key,
                         aws_secret_access_key=aws_secret)
        
        # Check MLflow artifacts in S3
        print("\n📦 Checking S3 for MLflow artifacts...")
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix="mlflow/")
        
        if 'Contents' in response:
            print(f"✅ Found {len(response['Contents'])} objects in S3:")
            for obj in response['Contents']:
                print(f"   📄 {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("ℹ️  No MLflow artifacts found in S3 yet")
            
    except Exception as e:
        print(f"❌ S3 check failed: {e}")

if __name__ == "__main__":
    test_pipeline_s3_artifacts()