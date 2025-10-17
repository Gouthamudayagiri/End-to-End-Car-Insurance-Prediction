# dvc_commands.py
import subprocess
import sys

def dvc_status():
    """Check DVC status"""
    result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0

def dvc_push():
    """Push DVC data to remote storage"""
    result = subprocess.run(["dvc", "push"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ DVC data pushed to S3 successfully")
    else:
        print(f"❌ DVC push failed: {result.stderr}")
    return result.returncode == 0

def dvc_pull():
    """Pull DVC data from remote storage"""
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ DVC data pulled from S3 successfully")
    else:
        print(f"❌ DVC pull failed: {result.stderr}")
    return result.returncode == 0

def show_metrics():
    """Show DVC metrics"""
    result = subprocess.run(["dvc", "metrics", "show"], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            dvc_status()
        elif command == "push":
            dvc_push()
        elif command == "pull":
            dvc_pull()
        elif command == "metrics":
            show_metrics()
        else:
            print("Usage: python dvc_commands.py [status|push|pull|metrics]")
    else:
        print("Available commands: status, push, pull, metrics")