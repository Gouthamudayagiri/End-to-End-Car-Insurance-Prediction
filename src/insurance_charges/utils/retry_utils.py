import time
import logging
from functools import wraps
from src.insurance_charges.exception import InsuranceException
import sys

def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_attempt = 0
            current_delay = delay
            
            while current_attempt < attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    current_attempt += 1
                    if current_attempt == attempts:
                        logging.error(f"All {attempts} attempts failed for {func.__name__}")
                        raise InsuranceException(e, sys)
                    
                    logging.warning(f"Attempt {current_attempt} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator