import os
import sys
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging

from src.insurance_charges.components.data_ingestion import DataIngestion
from src.insurance_charges.components.data_validation import DataValidation
from src.insurance_charges.components.data_transformation import DataTransformation
from src.insurance_charges.components.model_trainer import ModelTrainer
from src.insurance_charges.components.model_evaluation import ModelEvaluation
from src.insurance_charges.components.model_pusher import ModelPusher
from src.insurance_charges.components.data_analysis import DataAnalysis

from src.insurance_charges.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from src.insurance_charges.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

from src.insurance_charges.constants import SCHEMA_FILE_PATH
from src.insurance_charges.utils.main_utils import load_environment_variables, validate_environment_variables


class TrainPipeline:
    def __init__(self):
        """
        Initialize training pipeline with all configuration objects
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig()
            self.model_pusher_config = ModelPusherConfig()
            logging.info("Training pipeline initialized successfully")
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def validate_pipeline_config(self) -> bool:
        """
        Validate all pipeline configurations before execution
        """
        try:
            logging.info("Validating pipeline configurations...")
            
            # Validate data ingestion config
            assert self.data_ingestion_config.train_test_split_ratio > 0, "Train-test split ratio must be greater than 0"
            assert self.data_ingestion_config.train_test_split_ratio < 1, "Train-test split ratio must be less than 1"
            
            # Validate model trainer config
            assert 0 <= self.model_trainer_config.expected_accuracy <= 1, "Expected accuracy must be between 0 and 1"
            
            # Validate file paths exist
            assert os.path.exists(self.model_trainer_config.model_config_file_path), f"Model config file not found: {self.model_trainer_config.model_config_file_path}"
            assert os.path.exists(SCHEMA_FILE_PATH), f"Schema file not found: {SCHEMA_FILE_PATH}"
            
            logging.info("✅ All pipeline configurations are valid")
            return True
            
        except Exception as e:
            logging.error(f"❌ Pipeline configuration validation failed: {e}")
            raise InsuranceException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start data ingestion component
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Ingestion...")
            logging.info("=" * 50)
            
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info("✅ Data ingestion completed successfully")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_data_analysis(self, data_ingestion_artifact: DataIngestionArtifact) -> dict:
        """
        Perform comprehensive data analysis
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Analysis...")
            logging.info("=" * 50)
            
            # Create analysis artifact directory
            analysis_artifact_dir = os.path.join(
                self.data_ingestion_config.data_ingestion_dir, 
                'analysis_artifacts'
            )
            
            data_analysis = DataAnalysis(
                data_ingestion_artifact=data_ingestion_artifact,
                artifact_dir=analysis_artifact_dir
            )
            analysis_report = data_analysis.perform_analysis()
            
            logging.info("✅ Data analysis completed successfully")
            logging.info(f"Analysis artifacts saved to: {analysis_artifact_dir}")
            return analysis_report
            
        except Exception as e:
            logging.warning(f"Data analysis failed, but continuing pipeline: {e}")
            return {"error": str(e)}

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Start data validation component
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Validation...")
            logging.info("=" * 50)
            
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            
            if not data_validation_artifact.validation_status:
                logging.warning(f"Data validation failed: {data_validation_artifact.message}")
            else:
                logging.info("✅ Data validation completed successfully")
                
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_data_transformation(self, 
                                data_ingestion_artifact: DataIngestionArtifact, 
                                data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Start data transformation component
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Transformation...")
            logging.info("=" * 50)
            
            # Only proceed if data validation passed
            if not data_validation_artifact.validation_status:
                raise InsuranceException(
                    f"Cannot proceed with data transformation. Data validation failed: {data_validation_artifact.message}", 
                    sys
                )
            
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            logging.info("✅ Data transformation completed successfully")
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Start model training component
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Training...")
            logging.info("=" * 50)
            
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logging.info("✅ Model training completed successfully")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_model_evaluation(self, 
                             data_ingestion_artifact: DataIngestionArtifact,
                             model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        Start model evaluation component
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Evaluation...")
            logging.info("=" * 50)
            
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            
            logging.info("✅ Model evaluation completed successfully")
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Start model pushing component
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Model Pushing...")
            logging.info("=" * 50)
            
            # Only push if model is accepted
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted, skipping model pushing")
                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path
                )
            
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            
            logging.info("✅ Model pushing completed successfully")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Execute the complete training pipeline
        """
        try:
            logging.info("=" * 80)
            logging.info("🚀 STARTING INSURANCE CHARGES TRAINING PIPELINE")
            logging.info("=" * 80)
            
            # Step 1: Load environment variables (but don't require AWS for local development)
            logging.info("Step 1: Loading environment variables...")
            load_environment_variables()
            
            # Step 2: Validate configurations (skip AWS validation for now)
            logging.info("Step 2: Validating pipeline configurations...")
            self.validate_pipeline_config()
            
            # Step 3: Data Ingestion
            logging.info("Step 3: Starting data ingestion...")
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 4: Data Analysis
            logging.info("Step 4: Performing data analysis...")
            analysis_report = self.start_data_analysis(data_ingestion_artifact)
            logging.info(f"Data Analysis Summary: {analysis_report.get('quality_score', 'N/A')}")
            
            # Step 5: Data Validation
            logging.info("Step 5: Starting data validation...")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            
            # Step 6: Data Transformation
            logging.info("Step 6: Starting data transformation...")
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact, 
                data_validation_artifact
            )
            
            # Step 7: Model Training
            logging.info("Step 7: Starting model training...")
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            
            # Step 8: Model Evaluation
            logging.info("Step 8: Starting model evaluation...")
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact,
                model_trainer_artifact
            )
            
            # Step 9: Model Pushing (Conditional - handle AWS errors gracefully)
            if model_evaluation_artifact.is_model_accepted:
                logging.info("Step 9: Attempting model deployment...")
                try:
                    # Check if AWS credentials are valid
                    from src.insurance_charges.cloud_storage.aws_storage import SimpleStorageService
                    s3_client = SimpleStorageService()
                    
                    # Try to list buckets to validate credentials
                    s3_client.s3_client.list_buckets()
                    
                    # If we get here, AWS credentials are valid
                    model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
                    logging.info(f"✅ Model deployed to S3: {model_pusher_artifact.s3_model_path}")
                    
                except Exception as aws_error:
                    logging.warning(f"⚠️ AWS deployment skipped: {aws_error}")
                    logging.info("🎯 Model trained successfully and saved locally!")
                    logging.info(f"📍 Local model path: {model_evaluation_artifact.trained_model_path}")
            else:
                logging.info("Step 9: Model not accepted, skipping deployment")
            
            logging.info("=" * 80)
            logging.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error("=" * 80)
            logging.error("❌ TRAINING PIPELINE FAILED!")
            logging.error(f"Error: {e}")
            logging.error("=" * 80)
            raise InsuranceException(e, sys) from e