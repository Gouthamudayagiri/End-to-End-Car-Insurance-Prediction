# test_s3_connection.py
import os
import boto3
from dotenv import load_dotenv

def test_s3_connection():
    """Test if we can connect to S3 and access the bucket"""
    load_dotenv()
    
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = "insurance-charges-model-2025"
    
    print("🔍 Testing S3 Connection...")
    print(f"AWS_ACCESS_KEY_ID: {aws_key[:10]}..." if aws_key else "❌ AWS_ACCESS_KEY_ID not set")
    print(f"AWS_SECRET_ACCESS_KEY: {aws_secret[:10]}..." if aws_secret else "❌ AWS_SECRET_ACCESS_KEY not set")
    
    if not aws_key or not aws_secret:
        print("❌ AWS credentials missing - check your .env file")
        return False
    
    try:
        # Test S3 connection
        s3 = boto3.client('s3', 
                         aws_access_key_id=aws_key,
                         aws_secret_access_key=aws_secret)
        
        # List buckets to test connection
        response = s3.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        print(f"✅ Connected to AWS S3 - {len(buckets)} buckets found")
        
        # Check if our bucket exists
        if bucket_name in buckets:
            print(f"✅ Bucket '{bucket_name}' exists")
            
            # Test bucket permissions
            try:
                s3.head_bucket(Bucket=bucket_name)
                print(f"✅ Bucket '{bucket_name}' is accessible")
                
                # Test writing a small file
                test_key = "test-connection.txt"
                s3.put_object(Bucket=bucket_name, Key=test_key, Body=b"Test connection successful")
                print(f"✅ Successfully wrote test file to s3://{bucket_name}/{test_key}")
                
                # Clean up
                s3.delete_object(Bucket=bucket_name, Key=test_key)
                print(f"✅ Test file cleaned up")
                
                return True
                
            except Exception as e:
                print(f"❌ Bucket access failed: {e}")
                return False
        else:
            print(f"❌ Bucket '{bucket_name}' does not exist")
            print(f"💡 Create it with: aws s3 mb s3://{bucket_name}")
            return False
            
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False

if __name__ == "__main__":
    test_s3_connection()