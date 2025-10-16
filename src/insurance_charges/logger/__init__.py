import logging
import os
import sys
from from_root import from_root
from datetime import datetime

# Create log directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(from_root(), log_dir, LOG_FILE)

# Log format
LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# Configure logging
logging.basicConfig(
    filename=logs_path,
    format=LOG_FORMAT,
    level=logging.INFO,
    # Add stream handler for console output
    handlers=[
        logging.FileHandler(logs_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger instance
logger = logging.getLogger("insurance_charges")

def get_logger():
    return logger