# test_env.py
import os
from dotenv import load_dotenv

print("🔍 Testing .env file loading...")

# Load environment variables
load_dotenv()

# Check specific variables
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
postgres_url = os.getenv('POSTGRES_URL')

print(f"AWS_ACCESS_KEY_ID: {'✅ Found' if aws_key else '❌ Missing'}")
print(f"AWS_SECRET_ACCESS_KEY: {'✅ Found' if aws_secret else '❌ Missing'}")
print(f"POSTGRES_URL: {'✅ Found' if postgres_url else '❌ Missing'}")

if aws_key and aws_secret:
    print(f"AWS Key starts with: {aws_key[:10]}...")
    print(f"AWS Secret starts with: {aws_secret[:10]}...")