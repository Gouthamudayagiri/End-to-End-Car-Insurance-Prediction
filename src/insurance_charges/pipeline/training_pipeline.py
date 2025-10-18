import os
import sys
import datetime
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
from src.insurance_charges.utils.main_utils import load_environment_variables, validate_environment_variables, load_object

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    from src.insurance_charges.utils.mlflow_config import MLflowConfig
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available - proceeding without experiment tracking")


class TrainPipeline:
    def __init__(self):
        """
        Initialize training pipeline with dual storage configuration
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig()
            self.model_pusher_config = ModelPusherConfig()
            
            # DUAL STORAGE MLflow initialization
            if MLFLOW_AVAILABLE:
                # Clean up any active runs first
                if mlflow.active_run():
                    mlflow.end_run()
                
                self.mlflow_config = MLflowConfig()
                storage_info = self.mlflow_config.get_storage_info()
                
                logging.info(f"✅ MLflow configured with DUAL storage")
                logging.info(f"📊 Tracking: {storage_info['tracking_uri']}")
                logging.info(f"🎯 Experiment: {storage_info['experiment_name']}")
                logging.info(f"💾 Local: {storage_info['local_enabled']}")
                logging.info(f"☁️  S3: {storage_info['s3_enabled']}")
            else:
                logging.info("📊 MLflow not available - using local training only")
                
            logging.info("✅ Training pipeline initialized successfully")
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def validate_pipeline_config(self) -> bool:
        """
        Validate all pipeline configurations before execution
        """
        try:
            logging.info("🔍 Validating pipeline configurations...")
            
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
            logging.info("📥 Starting Data Ingestion...")
            logging.info("=" * 50)
            
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info("✅ Data ingestion completed successfully")
            logging.info(f"📦 Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_data_analysis(self, data_ingestion_artifact: DataIngestionArtifact) -> dict:
        """
        Perform comprehensive data analysis
        """
        try:
            logging.info("=" * 50)
            logging.info("📊 Starting Data Analysis...")
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
            logging.info(f"📊 Analysis artifacts saved to: {analysis_artifact_dir}")
            return analysis_report
            
        except Exception as e:
            logging.warning(f"⚠️ Data analysis failed, but continuing pipeline: {e}")
            return {"error": str(e)}

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Start data validation component
        """
        try:
            logging.info("=" * 50)
            logging.info("🔍 Starting Data Validation...")
            logging.info("=" * 50)
            
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            
            if not data_validation_artifact.validation_status:
                logging.warning(f"⚠️ Data validation failed: {data_validation_artifact.message}")
            else:
                logging.info("✅ Data validation completed successfully")
                
            logging.info(f"📦 Data validation artifact: {data_validation_artifact}")
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
            logging.info("🔄 Starting Data Transformation...")
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
            logging.info(f"📦 Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Start model training component with DUAL MLflow tracking
        """
        try:
            logging.info("=" * 50)
            logging.info("🤖 Starting Model Training...")
            logging.info("=" * 50)
            
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            # Log training metrics and artifacts to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                try:
                    # Log metrics
                    mlflow.log_metrics({
                        "train_r2_score": float(model_trainer_artifact.metric_artifact.r2_score),
                        "train_rmse": float(model_trainer_artifact.metric_artifact.rmse),
                        "train_mae": float(model_trainer_artifact.metric_artifact.mae)
                    })
                    
                    # Log parameters
                    mlflow.log_params({
                        "best_model": model_trainer_artifact.model_name,
                        "feature_count": model_trainer_artifact.feature_count,
                        "model_accuracy": float(model_trainer_artifact.metric_artifact.r2_score)
                    })
                    
                    # Log the trained model to BOTH local and S3
                    if hasattr(model_trainer_artifact, 'trained_model') and model_trainer_artifact.trained_model is not None:
                        mlflow.sklearn.log_model(
                            model_trainer_artifact.trained_model,
                            "trained_model",
                            registered_model_name=f"insurance_charges_{model_trainer_artifact.model_name}"
                        )
                        logging.info("✅ Trained model logged to MLflow (dual storage)")
                    
                    # Log preprocessing object
                    preprocessing_obj = load_object(data_transformation_artifact.transformed_object_file_path)
                    mlflow.sklearn.log_model(
                        preprocessing_obj,
                        "preprocessor"
                    )
                    logging.info("✅ Preprocessing object logged to MLflow (dual storage)")
                    
                    # Log model report using dual storage
                    model_report_path = os.path.join(
                        os.path.dirname(model_trainer_artifact.trained_model_file_path),
                        'artifacts',
                        'model_report.yaml'
                    )
                    if os.path.exists(model_report_path):
                        self.mlflow_config.log_artifacts_dual(
                            model_report_path, 
                            "model_report"
                        )
                        logging.info("✅ Model report logged with dual storage")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Could not log all artifacts to MLflow: {e}")
                
                logging.info("✅ Training metrics and artifacts logged with DUAL storage")
            
            logging.info("✅ Model training completed successfully")
            logging.info(f"📦 Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_model_evaluation(self, 
                             data_ingestion_artifact: DataIngestionArtifact,
                             model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        Start model evaluation component with DUAL MLflow tracking
        """
        try:
            logging.info("=" * 50)
            logging.info("📊 Starting Model Evaluation...")
            logging.info("=" * 50)
            
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            
            # Log evaluation results to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_metrics({
                    "test_r2_score": getattr(model_evaluation_artifact, 'changed_accuracy', 0),
                    "is_model_accepted": int(model_evaluation_artifact.is_model_accepted)
                })
                
                # Sync evaluation artifacts to S3
                if hasattr(self, 'mlflow_config') and self.mlflow_config.s3_enabled:
                    self.mlflow_config.sync_artifacts_to_s3()
                
                logging.info("✅ Evaluation metrics logged with dual storage")
            
            logging.info("✅ Model evaluation completed successfully")
            logging.info(f"📦 Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Start model pushing component with DUAL storage
        """
        try:
            logging.info("=" * 50)
            logging.info("🚀 Starting Model Pushing...")
            logging.info("=" * 50)
            
            # Only push if model is accepted
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("📭 Model not accepted, skipping model pushing")
                
                # Log to MLflow
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_param("model_pushed", False)
                    mlflow.log_param("push_reason", "model_not_accepted")
                    
                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path
                )
            
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            
            # Log successful push to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_param("model_pushed", True)
                mlflow.log_param("s3_location", model_pusher_artifact.s3_model_path)
                
                # Ensure artifacts are synced to S3
                if hasattr(self, 'mlflow_config') and self.mlflow_config.s3_enabled:
                    self.mlflow_config.sync_artifacts_to_s3()
                
                logging.info("✅ Model push status logged with dual storage")
            
            logging.info("✅ Model pushing completed successfully")
            logging.info(f"📦 Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def ensure_dual_storage_sync(self):
        """Ensure artifacts are synced to both local and S3 storage"""
        try:
            if not MLFLOW_AVAILABLE or not mlflow.active_run():
                return False
                
            # Use MLflow config for dual storage sync
            if hasattr(self, 'mlflow_config'):
                return self.mlflow_config.sync_artifacts_to_s3()
            else:
                return False
                
        except Exception as e:
            logging.warning(f"⚠️ Dual storage sync failed: {e}")
            return False

    def run_pipeline(self) -> None:
        """
        Execute the complete training pipeline with DUAL storage tracking
        """
        try:
            logging.info("=" * 80)
            logging.info("🚀 STARTING INSURANCE CHARGES TRAINING PIPELINE")
            logging.info("=" * 80)
            
            # Step 1: Load environment variables
            logging.info("Step 1: Loading environment variables...")
            load_environment_variables()
            
            # Step 2: Validate configurations
            logging.info("Step 2: Validating pipeline configurations...")
            self.validate_pipeline_config()
            
            # Start main MLflow run for entire pipeline with DUAL storage
            if MLFLOW_AVAILABLE:
                import datetime
                # Ensure no active runs
                if mlflow.active_run():
                    mlflow.end_run()
                
                # Start main pipeline run
                pipeline_run = mlflow.start_run(
                    run_name=f"pipeline_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags={
                        "pipeline_type": "full_training", 
                        "project": "insurance_charges",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "storage_type": "dual_local_s3",
                        "s3_enabled": str(getattr(self.mlflow_config, 's3_enabled', False))
                    }
                )
                logging.info(f"📊 MLflow pipeline run started: {pipeline_run.info.run_id}")
                logging.info(f"📍 Run URI: {pipeline_run.info.artifact_uri}")
                
                # Log storage configuration
                if hasattr(self, 'mlflow_config'):
                    storage_info = self.mlflow_config.get_storage_info()
                    mlflow.log_params({
                        "local_storage": str(storage_info['local_enabled']),
                        "s3_storage": str(storage_info['s3_enabled']),
                        "s3_bucket": storage_info.get('s3_bucket', 'none')
                    })
                    logging.info("🎯 DUAL storage configuration logged")
            
            # Step 3: Data Ingestion
            logging.info("Step 3: Starting data ingestion...")
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Log data ingestion info to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_params({
                    "train_test_split": self.data_ingestion_config.train_test_split_ratio,
                    "dataset_size": getattr(data_ingestion_artifact, 'dataset_size', 'unknown')
                })
            
            # Step 4: Data Analysis
            logging.info("Step 4: Performing data analysis...")
            analysis_report = self.start_data_analysis(data_ingestion_artifact)
            logging.info(f"📊 Data Analysis Summary: {analysis_report.get('quality_score', 'N/A')}")
            
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
            
            # 🔄 CRITICAL: Force DUAL storage sync after model training
            if MLFLOW_AVAILABLE and mlflow.active_run():
                logging.info("🔄 Syncing artifacts to DUAL storage after model training...")
                self.ensure_dual_storage_sync()
            
            # Step 8: Model Evaluation
            logging.info("Step 8: Starting model evaluation...")
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact,
                model_trainer_artifact
            )
            
            # Step 9: Model Pushing (Conditional - with DUAL storage)
            if model_evaluation_artifact.is_model_accepted:
                logging.info("Step 9: Attempting model deployment with DUAL storage...")
                try:
                    # Check if AWS credentials are valid for S3 deployment
                    from src.insurance_charges.cloud_storage.aws_storage import SimpleStorageService
                    s3_client = SimpleStorageService()
                    
                    # Try to list buckets to validate credentials
                    s3_client.s3_client.list_buckets()
                    
                    # If we get here, AWS credentials are valid
                    model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
                    logging.info(f"✅ Model deployed to S3: {model_pusher_artifact.s3_model_path}")
                    
                except Exception as aws_error:
                    logging.warning(f"⚠️ AWS deployment skipped: {aws_error}")
                    logging.info("🎯 Model trained successfully and saved locally with DUAL MLflow tracking!")
                    logging.info(f"📍 Local model path: {model_evaluation_artifact.trained_model_path}")
                    
                    # Log deployment status to DUAL MLflow storage
                    if MLFLOW_AVAILABLE and mlflow.active_run():
                        mlflow.log_param("deployment_status", "local_only")
                        mlflow.log_param("deployment_error", str(aws_error))
            else:
                logging.info("Step 9: Model not accepted, skipping deployment")
                # Log to DUAL MLflow storage
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_param("deployment_status", "not_accepted")
            
            # Log final artifacts to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                try:
                    # Log analysis reports if they exist using dual storage
                    analysis_artifact_dir = os.path.join(
                        self.data_ingestion_config.data_ingestion_dir, 
                        'analysis_artifacts'
                    )
                    if os.path.exists(analysis_artifact_dir):
                        if hasattr(self, 'mlflow_config'):
                            self.mlflow_config.log_artifacts_dual(
                                analysis_artifact_dir, 
                                "data_analysis"
                            )
                        else:
                            mlflow.log_artifacts(analysis_artifact_dir, "data_analysis")
                        logging.info("✅ Data analysis artifacts logged with dual storage")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Could not log final artifacts: {e}")
            
            # Log final pipeline status to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_params({
                    "pipeline_status": "completed_successfully",
                    "total_steps_completed": "9",
                    "final_model_accepted": str(model_evaluation_artifact.is_model_accepted),
                    "storage_used": "dual_local_s3" if getattr(self.mlflow_config, 's3_enabled', False) else "local_only"
                })
                
                # Log final metrics summary
                if hasattr(model_trainer_artifact, 'metric_artifact'):
                    mlflow.log_metrics({
                        "final_r2_score": float(model_trainer_artifact.metric_artifact.r2_score),
                        "final_rmse": float(model_trainer_artifact.metric_artifact.rmse),
                        "final_mae": float(model_trainer_artifact.metric_artifact.mae)
                    })
                
                # 🔄 FINAL SYNC: Ensure all artifacts are synced to DUAL storage
                logging.info("🔄 Performing final DUAL storage sync...")
                self.ensure_dual_storage_sync()
                
                mlflow.end_run()
                logging.info("✅ MLflow pipeline run completed with DUAL storage")
            
            logging.info("=" * 80)
            logging.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY WITH DUAL STORAGE!")
            logging.info("=" * 80)
            
        except Exception as e:
            # Log pipeline failure to DUAL MLflow storage
            if MLFLOW_AVAILABLE and mlflow.active_run():
                try:
                    mlflow.log_param("pipeline_status", "failed")
                    mlflow.log_param("error", str(e))
                    
                    # 🔄 SYNC ON ERROR: Try to sync artifacts even if pipeline fails
                    logging.info("🔄 Syncing artifacts after pipeline failure...")
                    self.ensure_dual_storage_sync()
                    mlflow.end_run()
                except:
                    pass
            
            logging.error("=" * 80)
            logging.error("❌ TRAINING PIPELINE FAILED!")
            logging.error(f"Error: {e}")
            logging.error("=" * 80)
            raise InsuranceException(e, sys) from e