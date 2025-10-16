import os
from datetime import date

DATABASE_NAME = "INSURANCE_DB"
COLLECTION_NAME = "insurance_data"
POSTGRES_URL_KEY = "POSTGRES_URL"

PIPELINE_NAME: str = "insurance_charges"
ARTIFACT_DIR: str = "artifact"
MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "charges"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "insurance.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

# Data Ingestion
DATA_INGESTION_COLLECTION_NAME: str = "insurance_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model Trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
# Add these missing constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_PUSHER_DIR_NAME: str = "model_pusher"

# Model configuration
HYPERPARAMETER_CONFIG_FILE_PATH: str = os.path.join("config", "hyperparameters.yaml")

# Database configuration
POSTGRES_SCHEMA: str = "public"  # Add schema support

# Feature store
FEATURE_STORE_DIR: str = "feature_store"
# Model Evaluation
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "insurance-charges-model"
MODEL_PUSHER_S3_KEY = "model-registry"

APP_HOST = "0.0.0.0"
APP_PORT = 8080

import os
TABLE_NAME = os.getenv("TABLE_NAME", "insurance_data")
SCHEMA_NAME = os.getenv("SCHEMA_NAME", "public")