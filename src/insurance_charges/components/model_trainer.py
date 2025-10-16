import os
import json
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from neuro_mf import ModelFactory

from insurance_charges.exception import InsuranceException
from insurance_charges.logger import logging
from insurance_charges.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object, write_yaml_file
from insurance_charges.entity.config_entity import ModelTrainerConfig
from insurance_charges.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from insurance_charges.entity.estimator import InsuranceModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model trainer
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.model_artifacts_dir = os.path.join(os.path.dirname(model_trainer_config.trained_model_file_path), 'artifacts')
        os.makedirs(self.model_artifacts_dir, exist_ok=True)

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object, dict]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            # Make predictions
            y_pred_train = model_obj.predict(x_train)
            y_pred_test = model_obj.predict(x_test)
            
            # Calculate regression metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            
            # Create detailed model report
            model_report = {
                'best_model': type(model_obj).__name__,
                'best_score': best_model_detail.best_score,
                'model_parameters': best_model_detail.best_parameters,
                'performance_metrics': {
                    'train_r2_score': float(train_r2),
                    'test_r2_score': float(test_r2),
                    'test_rmse': float(rmse),
                    'test_mae': float(mae)
                },
                'cross_validation_scores': [float(score) for score in best_model_detail.cross_validation_scores] if hasattr(best_model_detail, 'cross_validation_scores') else [],
                'all_models_tried': [
                    {
                        'model_name': model.model_name,
                        'model_score': model.model_score,
                        'model_parameters': model.parameters
                    } for model in best_model_detail.all_models
                ] if hasattr(best_model_detail, 'all_models') else []
            }
            
            metric_artifact = RegressionMetricArtifact(
                r2_score=test_r2, 
                rmse=rmse, 
                mae=mae
            )
            
            return best_model_detail, metric_artifact, model_report
        
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail, metric_artifact, model_report = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            insurance_model = InsuranceModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )
            
            logging.info("Created insurance model object with preprocessor and model")
            
            # Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, insurance_model)
            
            # Save model report as artifact
            model_report_path = os.path.join(self.model_artifacts_dir, 'model_report.yaml')
            write_yaml_file(model_report_path, model_report)
            
            # Save feature importance if available
            if hasattr(best_model_detail.best_model, 'feature_importances_'):
                feature_importance = best_model_detail.best_model.feature_importances_
                feature_importance_dict = {
                    f'feature_{i}': float(importance) 
                    for i, importance in enumerate(feature_importance)
                }
                feature_importance_path = os.path.join(self.model_artifacts_dir, 'feature_importance.yaml')
                write_yaml_file(feature_importance_path, feature_importance_dict)
            
            # Save training history/curves if available
            training_history = {
                'best_score': float(best_model_detail.best_score),
                'expected_accuracy': float(self.model_trainer_config.expected_accuracy),
                'model_training_time': getattr(best_model_detail, 'training_time', 'Not available')
            }
            training_history_path = os.path.join(self.model_artifacts_dir, 'training_history.yaml')
            write_yaml_file(training_history_path, training_history)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info(f"Model artifacts saved to: {self.model_artifacts_dir}")
            
            return model_trainer_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e