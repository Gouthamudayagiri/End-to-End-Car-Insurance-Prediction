import sys
from insurance_charges.exception import InsuranceException
from insurance_charges.logger import logging

from insurance_charges.components.data_ingestion import DataIngestion
from insurance_charges.components.data_validation import DataValidation
from insurance_charges.components.data_transformation import DataTransformation
from insurance_charges.components.model_trainer import ModelTrainer
from insurance_charges.components.model_evaluation import ModelEvaluation
from insurance_charges.components.model_pusher import ModelPusher

from insurance_charges.entity.config_entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig,
                                          ModelEvaluationConfig,
                                          ModelPusherConfig)
                                          
from insurance_charges.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from PostgreSQL")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from PostgreSQL")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                                 data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting model evaluation
        """
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise InsuranceException(e, sys)
    def validate_pipeline_config(self):
        """
        Validate all pipeline configurations before execution
        """
        try:
            # Validate data ingestion config
            assert self.data_ingestion_config.train_test_split_ratio > 0
            assert self.data_ingestion_config.train_test_split_ratio < 1
            
            # Validate model trainer config
            assert 0 <= self.model_trainer_config.expected_accuracy <= 1
            
            # Validate file paths exist
            assert os.path.exists(self.model_trainer_config.model_config_file_path)
            assert os.path.exists(SCHEMA_FILE_PATH)
            
            logging.info("All pipeline configurations are valid")
            return True
            
        except Exception as e:
            logging.error(f"Pipeline configuration validation failed: {e}")
            raise InsuranceException(e, sys)

    # def run_pipeline(self) -> None:
    #     """
    #     This method of TrainPipeline class is responsible for running complete pipeline
    #     """
    #     try:
    #         data_ingestion_artifact = self.start_data_ingestion()
    #         data_validation_artifact = self.start_data_validation(
    #             data_ingestion_artifact=data_ingestion_artifact
    #         )
    #         data_transformation_artifact = self.start_data_transformation(
    #             data_ingestion_artifact=data_ingestion_artifact, 
    #             data_validation_artifact=data_validation_artifact
    #         )
    #         model_trainer_artifact = self.start_model_trainer(
    #             data_transformation_artifact=data_transformation_artifact
    #         )
    #         model_evaluation_artifact = self.start_model_evaluation(
    #             data_ingestion_artifact=data_ingestion_artifact,
    #             model_trainer_artifact=model_trainer_artifact
    #         )
            
    #         if not model_evaluation_artifact.is_model_accepted:
    #             logging.info(f"Model not accepted.")
    #             return None
                
    #         model_pusher_artifact = self.start_model_pusher(
    #             model_evaluation_artifact=model_evaluation_artifact
    #         )
    #         logging.info("Training pipeline completed successfully!")

    #     except Exception as e:
    #         raise InsuranceException(e, sys)
    def start_data_analysis(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        This method performs comprehensive data analysis
        """
        try:
            logging.info("Starting data analysis")
            from insurance_charges.components.data_analysis import DataAnalysis
            
            # Create analysis artifact directory
            analysis_artifact_dir = os.path.join(self.data_ingestion_config.data_ingestion_dir, 'analysis_artifacts')
            
            data_analysis = DataAnalysis(
                data_ingestion_artifact=data_ingestion_artifact,
                artifact_dir=analysis_artifact_dir
            )
            analysis_report = data_analysis.perform_analysis()
            
            logging.info(f"Data analysis completed: {analysis_report}")
            logging.info(f"Analysis artifacts saved to: {analysis_artifact_dir}")
            
            return analysis_report
            
        except Exception as e:
            raise InsuranceException(e, sys)


def run_pipeline(self) -> None:
    """
    This method of TrainPipeline class is responsible for running complete pipeline
    """
    try:
        # Load environment variables at the start
        from insurance_charges.utils.main_utils import load_environment_variables
        load_environment_variables()
        
        # Validate configurations first
        self.validate_pipeline_config()
        
        data_ingestion_artifact = self.start_data_ingestion()
        
        # Perform data analysis
        analysis_report = self.start_data_analysis(data_ingestion_artifact)
        logging.info(f"Data Analysis Report: {analysis_report}")
        
        data_validation_artifact = self.start_data_validation(
            data_ingestion_artifact=data_ingestion_artifact
        )
        
        # Continue with rest of the pipeline...
        data_transformation_artifact = self.start_data_transformation(
            data_ingestion_artifact=data_ingestion_artifact, 
            data_validation_artifact=data_validation_artifact
        )
        
        model_trainer_artifact = self.start_model_trainer(
            data_transformation_artifact=data_transformation_artifact
        )
        
        model_evaluation_artifact = self.start_model_evaluation(
            data_ingestion_artifact=data_ingestion_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        
        if not model_evaluation_artifact.is_model_accepted:
            logging.info(f"Model not accepted.")
            return None
            
        model_pusher_artifact = self.start_model_pusher(
            model_evaluation_artifact=model_evaluation_artifact
        )
        
        logging.info("Training pipeline completed successfully!")

    except Exception as e:
        raise InsuranceException(e, sys)