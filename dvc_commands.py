# dvc_commands.py - UPDATED FOR DUAL STORAGE
import subprocess
import sys
import os
from dotenv import load_dotenv

class DVCCommands:
    def __init__(self):
        load_dotenv()
        self.remote_name = "s3-storage"
        self.bucket_name = "insurance-charges-model-2025"
        
    def dvc_init(self):
        """Initialize DVC if not already initialized"""
        if not os.path.exists('.dvc'):
            print("🚀 Initializing DVC...")
            result = subprocess.run(["dvc", "init"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ DVC initialized successfully")
                return True
            else:
                print(f"❌ DVC initialization failed: {result.stderr}")
                return False
        else:
            print("✅ DVC already initialized")
            return True
    
    def setup_s3_remote(self):
        """Setup S3 remote for DVC"""
        try:
            # Check if remote exists
            result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
            if self.remote_name not in result.stdout:
                print(f"📦 Setting up DVC S3 remote: {self.remote_name}")
                
                remote_url = f"s3://{self.bucket_name}/dvc-storage"
                setup_cmd = [
                    'dvc', 'remote', 'add', 
                    '-d', self.remote_name, 
                    remote_url
                ]
                
                result = subprocess.run(setup_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ Could not setup S3 remote: {result.stderr}")
                    return False
                
                # Configure AWS credentials
                aws_key = os.getenv('AWS_ACCESS_KEY_ID')
                aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
                
                if aws_key and aws_secret:
                    subprocess.run([
                        'dvc', 'remote', 'modify', 
                        self.remote_name, 
                        'access_key_id', aws_key
                    ], check=True)
                    
                    subprocess.run([
                        'dvc', 'remote', 'modify', 
                        self.remote_name, 
                        'secret_access_key', aws_secret
                    ], check=True)
                    
                    print("✅ DVC S3 remote configured with credentials")
                else:
                    print("⚠️ AWS credentials not found, S3 remote setup incomplete")
                
                print(f"✅ DVC S3 remote configured: {remote_url}")
                return True
            else:
                print("✅ DVC S3 remote already configured")
                return True
                
        except Exception as e:
            print(f"❌ S3 remote setup failed: {e}")
            return False
    
    def dvc_status(self):
        """Check DVC status with dual storage info"""
        try:
            print("🔍 DVC Status (Dual Storage):")
            
            # Get DVC status
            result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
            print(result.stdout)
            
            # Show remote info
            print("\n💾 Storage Configuration:")
            remote_result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
            print(remote_result.stdout)
            
            return result.returncode == 0
        except Exception as e:
            print(f"❌ DVC status failed: {e}")
            return False
    
    def dvc_push(self):
        """Push DVC data to S3 remote with dual storage"""
        try:
            print("📤 Pushing DVC data to S3 remote...")
            result = subprocess.run(["dvc", "push", "-r", self.remote_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ DVC data pushed to S3 successfully")
                print("💾 Data available in both local and S3 storage")
                return True
            else:
                print(f"❌ DVC push failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ DVC push failed: {e}")
            return False
    
    def dvc_pull(self):
        """Pull DVC data from S3 remote"""
        try:
            print("📥 Pulling DVC data from S3 remote...")
            result = subprocess.run(["dvc", "pull", "-r", self.remote_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ DVC data pulled from S3 successfully")
                return True
            else:
                print(f"❌ DVC pull failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ DVC pull failed: {e}")
            return False
    
    def dvc_repro(self):
        """Reproduce DVC pipeline with dual storage"""
        try:
            print("🔄 Reproducing DVC pipeline with dual storage...")
            result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ DVC pipeline reproduced successfully")
                
                # Auto-push to S3 after reproduction
                print("📤 Auto-pushing to S3 remote...")
                push_result = subprocess.run(["dvc", "push", "-r", self.remote_name], capture_output=True, text=True)
                
                if push_result.returncode == 0:
                    print("✅ Pipeline artifacts synced to S3")
                else:
                    print("⚠️ Pipeline reproduced but S3 sync failed")
                
                return True
            else:
                print(f"❌ DVC reproduction failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ DVC reproduction failed: {e}")
            return False
    
    def show_metrics(self):
        """Show DVC metrics from dual storage"""
        try:
            print("📊 DVC Metrics (Dual Storage):")
            result = subprocess.run(["dvc", "metrics", "show"], capture_output=True, text=True)
            print(result.stdout)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Metrics show failed: {e}")
            return False
    
    def get_storage_info(self):
        """Get comprehensive storage information"""
        try:
            print("💾 DVC Dual Storage Information:")
            
            # Local storage info
            local_size = 0
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if '.dvc' in root or 'data' in root or 'models' in root:
                        file_path = os.path.join(root, file)
                        local_size += os.path.getsize(file_path)
            
            print(f"📍 Local Storage: {local_size / (1024*1024):.2f} MB")
            
            # Remote info
            remote_result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True)
            print(f"☁️  Remote Storage: {self.remote_name} (s3://{self.bucket_name}/dvc-storage)")
            
            return True
        except Exception as e:
            print(f"❌ Storage info failed: {e}")
            return False

def main():
    dvc = DVCCommands()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "init":
            dvc.dvc_init()
            dvc.setup_s3_remote()
        elif command == "status":
            dvc.dvc_status()
        elif command == "push":
            dvc.dvc_push()
        elif command == "pull":
            dvc.dvc_pull()
        elif command == "repro":
            dvc.dvc_repro()
        elif command == "metrics":
            dvc.show_metrics()
        elif command == "storage":
            dvc.get_storage_info()
        elif command == "setup":
            dvc.dvc_init()
            dvc.setup_s3_remote()
            print("✅ DVC dual storage setup complete!")
        else:
            print("""
Usage: python dvc_commands.py [command]

Commands:
  init     - Initialize DVC
  setup    - Complete DVC dual storage setup
  status   - Check DVC status with storage info
  push     - Push data to S3 remote
  pull     - Pull data from S3 remote
  repro    - Reproduce pipeline with auto-push
  metrics  - Show DVC metrics
  storage  - Show storage information
            """)
    else:
        print("""
🎯 DVC Dual Storage Manager

This utility manages DVC with both local and S3 storage.

Quick Start:
1. python dvc_commands.py setup    # Initialize DVC + S3 remote
2. python dvc_pipeline.py --stage full  # Run pipeline
3. python dvc_commands.py push     # Push to S3

View Results:
- dvc metrics show                 # View metrics
- dvc status                       # Check status
- python dvc_commands.py storage   # Storage info
        """)

if __name__ == "__main__":
    main()