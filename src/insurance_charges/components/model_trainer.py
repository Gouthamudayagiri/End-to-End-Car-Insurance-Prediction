# src/insurance_charges/components/model_trainer.py
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from src.insurance_charges.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object, write_yaml_file
from src.insurance_charges.entity.config_entity import ModelTrainerConfig
from src.insurance_charges.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from src.insurance_charges.entity.estimator import InsuranceModel

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

    def get_best_model(self, train: np.array, test: np.array) -> tuple:
        """
        Find the best model using GridSearchCV without neuro-mf
        """
        try:
            logging.info("Starting model selection with GridSearchCV")
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            
            # Define models and their parameter grids
            models = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'xgboost': {
                    'model': XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5],
                        'subsample': [0.8, 0.9]
                    }
                }
            }
            
            best_score = -np.inf
            best_model = None
            best_model_name = None
            best_params = None
            all_model_results = {}
            
            for model_name, model_config in models.items():
                logging.info(f"Training {model_name} with GridSearchCV")
                
                try:
                    grid_search = GridSearchCV(
                        model_config['model'],
                        model_config['params'],
                        cv=5,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid_search.fit(x_train, y_train)
                    
                    # Test performance
                    y_pred = grid_search.predict(x_test)
                    test_r2 = r2_score(y_test, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store results
                    all_model_results[model_name] = {
                        'best_cv_score': float(grid_search.best_score_),
                        'test_r2_score': float(test_r2),
                        'test_rmse': float(test_rmse),
                        'test_mae': float(test_mae),
                        'best_params': grid_search.best_params_
                    }
                    
                    logging.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}, Test Score: {test_r2:.4f}")
                    
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = grid_search.best_estimator_
                        best_model_name = model_name
                        best_params = grid_search.best_params_
                        
                except Exception as e:
                    logging.error(f"Error training {model_name}: {e}")
                    continue
            
            if best_model is None:
                raise Exception("No model could be trained successfully")
            
            # Create comprehensive model report
            model_report = {
                'best_model': best_model_name,
                'best_score': float(best_score),
                'best_params': best_params,
                'all_models_tried': list(models.keys()),
                'all_model_results': all_model_results,
                'model_selection_method': 'grid_search_cv'
            }
            
            logging.info(f"🎯 Best model: {best_model_name} with R2 score: {best_score:.4f}")
            
            return best_model, model_report
            
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
            
            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")
            
            best_model, model_report = self.get_best_model(train=train_arr, test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if model_report['best_score'] < self.model_trainer_config.expected_accuracy:
                error_msg = f"No best model found with score more than base score. Best: {model_report['best_score']}, Expected: {self.model_trainer_config.expected_accuracy}"
                logging.error(error_msg)
                raise Exception(error_msg)

            insurance_model = InsuranceModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )
            
            logging.info("Created insurance model object with preprocessor and model")
            
            # Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, insurance_model)
            
            # Calculate final metrics on full test set
            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            y_pred = best_model.predict(x_test)
            
            final_r2 = r2_score(y_test, y_pred)
            final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            final_mae = mean_absolute_error(y_test, y_pred)
            
            # Update model report with final metrics
            model_report['final_metrics'] = {
                'r2_score': float(final_r2),
                'rmse': float(final_rmse),
                'mae': float(final_mae)
            }
            
            # Save model report as artifact
            model_report_path = os.path.join(self.model_artifacts_dir, 'model_report.yaml')
            write_yaml_file(model_report_path, model_report)
            
            # Save detailed predictions for analysis
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'residual': y_test - y_pred
            })
            predictions_path = os.path.join(self.model_artifacts_dir, 'predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)
            
            metric_artifact = RegressionMetricArtifact(
                r2_score=final_r2, 
                rmse=final_rmse, 
                mae=final_mae
            )

            # Create model trainer artifact with ALL required attributes
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                model_name=model_report['best_model'],
                feature_count=x_test.shape[1],
                trained_model=best_model  # This fixes the MLflow logging issue
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info(f"Model artifacts saved to: {self.model_artifacts_dir}")
            logging.info(f"Final Model Performance - R2: {final_r2:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
            
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise InsuranceException(e, sys) from e