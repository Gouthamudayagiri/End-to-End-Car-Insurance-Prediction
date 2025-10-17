# src/insurance_charges/entity/artifact_entity.py
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class DataIngestionArtifact:
    trained_file_path: str 
    test_file_path: str 
    dataset_size: int = 0

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str 
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class RegressionMetricArtifact:
    r2_score: float
    rmse: float
    mae: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str 
    metric_artifact: RegressionMetricArtifact
    model_name: str = "unknown"
    feature_count: int = 0
    trained_model: Any = None  # ADD THIS LINE - FIXES THE ISSUE

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    s3_model_path: str 
    trained_model_path: str

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str