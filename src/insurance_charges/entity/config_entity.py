# src/insurance_charges/entity/config_entity.py
import os
from src.insurance_charges.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    table_name: str = TABLE_NAME
    schema_name: str = SCHEMA_NAME

    def __post_init__(self):
        # Create directories
        os.makedirs(self.data_ingestion_dir, exist_ok=True)
        feature_store_dir = os.path.dirname(self.feature_store_file_path)
        training_dir = os.path.dirname(self.training_file_path)
        testing_dir = os.path.dirname(self.testing_file_path)
        
        os.makedirs(feature_store_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)
        os.makedirs(testing_dir, exist_ok=True)

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

    def __post_init__(self):
        os.makedirs(self.data_validation_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.drift_report_file_path), exist_ok=True)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCSSING_OBJECT_FILE_NAME)

    def __post_init__(self):
        os.makedirs(self.data_transformation_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.transformed_train_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.transformed_test_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.transformed_object_file_path), exist_ok=True)

@dataclass
class DataAnalysisConfig:
    data_analysis_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_analysis")
    analysis_report_file_path: str = os.path.join(data_analysis_dir, "analysis_report.yaml")
    plots_dir: str = os.path.join(data_analysis_dir, "plots")

    def __post_init__(self):
        os.makedirs(self.data_analysis_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.analysis_report_file_path), exist_ok=True)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

    def __post_init__(self):
        os.makedirs(self.model_trainer_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.trained_model_file_path), exist_ok=True)

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class InsurancePredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME