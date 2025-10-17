# dvc_with_env.py
import os
import subprocess
import sys
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if AWS credentials are loaded
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_key or not aws_secret:
        print("❌ AWS credentials not found in .env file")
        print("Please check your .env file has:")
        print("AWS_ACCESS_KEY_ID=your_key")
        print("AWS_SECRET_ACCESS_KEY=your_secret")
        return
    
    print(f"✅ AWS Access Key: {aws_key[:10]}...")
    print(f"✅ AWS Secret Key: {aws_secret[:10]}...")
    
    # Run DVC push
    print("🚀 Running DVC push with loaded credentials...")
    result = subprocess.run(['dvc', 'push'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ DVC push successful!")
    else:
        print(f"❌ DVC push failed: {result.stderr}")

if __name__ == "__main__":
    main()