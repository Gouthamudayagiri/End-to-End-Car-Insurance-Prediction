import os
import sys
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from insurance_charges.entity.config_entity import ModelEvaluationConfig
from insurance_charges.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from insurance_charges.exception import InsuranceException
from insurance_charges.constants import TARGET_COLUMN
from insurance_charges.logger import logging
from insurance_charges.utils.main_utils import write_yaml_file
from typing import Optional
from insurance_charges.entity.s3_estimator import InsuranceEstimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float
    evaluation_metrics: dict

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.evaluation_artifacts_dir = os.path.join(
                os.path.dirname(model_trainer_artifact.trained_model_file_path), 
                '../model_evaluation_artifacts'
            )
            os.makedirs(self.evaluation_artifacts_dir, exist_ok=True)
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def get_best_model(self) -> Optional[InsuranceEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            insurance_estimator = InsuranceEstimator(bucket_name=bucket_name,
                                                    model_path=model_path)

            if insurance_estimator.is_model_present(model_path=model_path):
                return insurance_estimator
            return None
        except Exception as e:
            raise InsuranceException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # Get trained model metrics
            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2_score
            trained_model_rmse = self.model_trainer_artifact.metric_artifact.rmse
            trained_model_mae = self.model_trainer_artifact.metric_artifact.mae

            best_model_r2_score = None
            best_model_rmse = None
            best_model_mae = None
            
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_r2_score = r2_score(y, y_hat_best_model)
                best_model_rmse = np.sqrt(mean_squared_error(y, y_hat_best_model))
                best_model_mae = mean_absolute_error(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score
            
            evaluation_metrics = {
                'trained_model': {
                    'r2_score': float(trained_model_r2_score),
                    'rmse': float(trained_model_rmse),
                    'mae': float(trained_model_mae)
                },
                'best_model': {
                    'r2_score': float(best_model_r2_score) if best_model_r2_score else None,
                    'rmse': float(best_model_rmse) if best_model_rmse else None,
                    'mae': float(best_model_mae) if best_model_mae else None
                } if best_model else None,
                'improvement': float(trained_model_r2_score - tmp_best_model_score) if best_model else None
            }
            
            result = EvaluateModelResponse(
                trained_model_r2_score=trained_model_r2_score,
                best_model_r2_score=best_model_r2_score,
                is_model_accepted=trained_model_r2_score > tmp_best_model_score,
                difference=trained_model_r2_score - tmp_best_model_score,
                evaluation_metrics=evaluation_metrics
            )
            
            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            # Save evaluation report as artifact
            evaluation_report = {
                'model_comparison': evaluate_model_response.evaluation_metrics,
                'decision': {
                    'is_model_accepted': evaluate_model_response.is_model_accepted,
                    'reason': 'New model performs better' if evaluate_model_response.is_model_accepted else 'Existing model performs better',
                    'accuracy_difference': float(evaluate_model_response.difference)
                },
                'threshold_config': {
                    'changed_threshold_score': float(self.model_eval_config.changed_threshold_score)
                }
            }
            
            evaluation_report_path = os.path.join(self.evaluation_artifacts_dir, 'evaluation_report.yaml')
            write_yaml_file(evaluation_report_path, evaluation_report)
            
            # Save detailed comparison
            comparison_details = {
                'trained_model_performance': {
                    'r2_score': float(evaluate_model_response.trained_model_r2_score),
                    'performance_status': 'ACCEPTED' if evaluate_model_response.is_model_accepted else 'REJECTED'
                },
                'production_model_performance': {
                    'r2_score': float(evaluate_model_response.best_model_r2_score) if evaluate_model_response.best_model_r2_score else 'No production model',
                    'performance_status': 'CURRENT_BEST' if not evaluate_model_response.is_model_accepted else 'REPLACED'
                } if evaluate_model_response.best_model_r2_score else None
            }
            
            comparison_path = os.path.join(self.evaluation_artifacts_dir, 'model_comparison.yaml')
            write_yaml_file(comparison_path, comparison_details)

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info(f"Evaluation artifacts saved to: {self.evaluation_artifacts_dir}")
            
            return model_evaluation_artifact
        except Exception as e:
            raise InsuranceException(e, sys) from e