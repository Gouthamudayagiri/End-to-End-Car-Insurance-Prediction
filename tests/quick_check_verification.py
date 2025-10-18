# quick_s3_check.py
import os
import subprocess

def check_s3_contents():
    """Check S3 bucket contents"""
    try:
        # Use Windows-compatible command
        cmd = 'aws s3 ls s3://insurance-charges-model-2025/ --recursive --human-readable'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = [line for line in result.stdout.split('\n') if line.strip()]
            print(f"✅ S3 bucket contains {len(lines)} items")
            
            # Show MLflow-related items
            mlflow_items = [line for line in lines if 'mlflow' in line]
            print(f"📊 MLflow items in S3: {len(mlflow_items)}")
            
            for item in mlflow_items[:5]:  # Show first 5
                print(f"  {item}")
                
            if len(mlflow_items) > 5:
                print(f"  ... and {len(mlflow_items) - 5} more")
        else:
            print("❌ Could not access S3 bucket")
            
    except Exception as e:
        print(f"❌ S3 check failed: {e}")

def check_local_mlflow():
    """Check local MLflow data"""
    if os.path.exists("mlruns"):
        file_count = 0
        for root, dirs, files in os.walk("mlruns"):
            file_count += len(files)
        print(f"✅ Local MLflow files: {file_count}")
    else:
        print("❌ No local mlruns directory")

print("=== QUICK S3 STATUS ===")
check_s3_contents()
print("\n=== LOCAL MLFLOW ===")
check_local_mlflow()
print("\n🎯 Run the permanent fix script to enable S3 for future runs!")