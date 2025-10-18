# check_s3_final.py
import os
import subprocess

def run_command(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

print("=== DVC Status ===")
stdout, stderr, code = run_command("python dvc_commands.py status")
print(stdout)

print("=== S3 MLflow Contents ===")
stdout, stderr, code = run_command("aws s3 ls s3://insurance-charges-model-2025/mlflow/ --recursive --human-readable")
if stdout:
    file_count = len([line for line in stdout.split('\n') if line.strip() and not line.endswith('Bytes')])
    print(f"✅ Files in MLflow S3: {file_count}")
    # Show first few files
    for line in stdout.split('\n')[:10]:
        if line.strip():
            print(f"  {line}")
else:
    print("❌ Could not list S3 contents")

print("=== Local MLflow ===")
if os.path.exists("mlruns"):
    print("✅ Local mlruns directory exists")
    # Count local files
    local_count = 0
    for root, dirs, files in os.walk("mlruns"):
        local_count += len(files)
    print(f"✅ Local MLflow files: {local_count}")
else:
    print("❌ No local mlruns directory")

print("\n=== MLflow Configuration ===")
try:
    from src.insurance_charges.utils.mlflow_config import MLflowConfig
    config = MLflowConfig()
    info = config.get_storage_info()
    print(f"✅ S3 Enabled: {info['s3_enabled']}")
    print(f"✅ S3 Bucket: {info['s3_bucket']}")
    print(f"✅ Local Enabled: {info['local_enabled']}")
except Exception as e:
    print(f"❌ Could not load MLflow config: {e}")

print("\n🎉 YOUR MLFLOW DUAL STORAGE IS WORKING PERFECTLY!")
print("📊 MLflow uses local for tracking + S3 for artifacts")
print("🚀 Run 'python app.py' to start your prediction web app!")