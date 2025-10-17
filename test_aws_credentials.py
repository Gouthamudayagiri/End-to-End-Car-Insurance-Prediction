# test_aws.py
import os
import boto3

def test_aws_credentials():
    print("🔍 Testing AWS Credentials...")
    
    # Check environment variables
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not access_key or not secret_key:
        print("❌ AWS credentials not found in environment variables")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    print(f"✅ AWS Access Key ID: {access_key[:10]}...")
    print(f"✅ AWS Secret Access Key: {secret_key[:10]}...")
    
    try:
        # Test S3 access
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        print("✅ Successfully connected to AWS S3")
        print(f"✅ Available buckets: {len(response['Buckets'])}")
        
        # Test your specific bucket
        try:
            s3.head_bucket(Bucket='insurance-charges-model-2025')
            print("✅ Bucket 'insurance-charges-model-2025' exists and is accessible")
        except:
            print("⚠️ Bucket 'insurance-charges-model-2025' doesn't exist or is not accessible")
            
        return True
        
    except Exception as e:
        print(f"❌ AWS connection failed: {e}")
        return False

if __name__ == "__main__":
    test_aws_credentials()