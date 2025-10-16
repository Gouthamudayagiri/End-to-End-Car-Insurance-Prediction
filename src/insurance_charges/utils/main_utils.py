# src/insurance_charges/utils/main_utils.py
import os
import sys
import numpy as np
import dill
import yaml
from pandas import DataFrame
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise InsuranceException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise InsuranceException(e, sys) from e

def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info("Exited the load_object method of utils")
        return obj
    except Exception as e:
        raise InsuranceException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file with allow_pickle=True
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array, allow_pickle=True)
        logging.info(f"Saved numpy array to {file_path}")
    except Exception as e:
        raise InsuranceException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file with allow_pickle=True
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise InsuranceException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise InsuranceException(e, sys) from e

def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns method of utils")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info("Exited the drop_columns method of utils")
        return df
    except Exception as e:
        raise InsuranceException(e, sys) from e

def load_environment_variables():
    """
    Load environment variables from .env file
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logging.info("Environment variables loaded successfully")
    except Exception as e:
        logging.warning(f"Could not load .env file: {e}")

def validate_environment_variables():
    """
    Validate that all required environment variables are set
    """
    required_vars = [
        "POSTGRES_URL",
        "AWS_ACCESS_KEY_ID", 
        "AWS_SECRET_ACCESS_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise InsuranceException(
            f"Missing required environment variables: {missing_vars}. ",
            sys
        )
    
    logging.info("All required environment variables are set")