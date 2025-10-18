# src/insurance_charges/components/data_validation.py
import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from src.insurance_charges.utils.main_utils import read_yaml_file, write_yaml_file
from src.insurance_charges.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.insurance_charges.entity.config_entity import DataValidationConfig
from src.insurance_charges.constants import SCHEMA_FILE_PATH
import evidently

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsuranceException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise InsuranceException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0 else True
        except Exception as e:
            raise InsuranceException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise InsuranceException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected
        """
        try:
            # Check if evidently is available
            evidently_available = False
            try:
                from evidently.report import Report
                from evidently.metrics import DatasetDriftMetric
                evidently_available = True
                logging.info("✅ Evidently available for drift detection")
            except ImportError:
                logging.warning("⚠️ Evidently not available. Using basic statistical drift detection.")
                evidently_available = False
            
            if evidently_available:
                return self._detect_drift_with_evidently(reference_df, current_df)
            else:
                return self._basic_statistical_drift(reference_df, current_df)
                
        except Exception as e:
            logging.error(f"❌ Drift detection failed: {e}")
            # Fallback to basic method
            return self._basic_statistical_drift(reference_df, current_df)

    def _detect_drift_with_evidently(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detect drift using Evidently AI
        """
        try:
            from evidently.report import Report
            from evidently.metrics import DatasetDriftMetric
            
            logging.info("🔍 Running Evidently drift detection...")
            
            # Ensure data types match schema
            reference_df = self._ensure_correct_data_types(reference_df)
            current_df = self._ensure_correct_data_types(current_df)
            
            # Create and run drift report
            drift_report = Report(metrics=[DatasetDriftMetric()])
            drift_report.run(reference_data=reference_df, current_data=current_df)
            
            # Get results
            report_result = drift_report.as_dict()
            
            # Extract drift information
            drift_detected = False
            n_features = 0
            n_drifted_features = 0
            
            if 'metrics' in report_result:
                for metric in report_result['metrics']:
                    if metric['metric'] == 'DatasetDriftMetric':
                        drift_detected = metric['result']['dataset_drift']
                        n_drifted_features = metric['result']['number_of_drifted_columns']
                        n_features = metric['result']['number_of_columns']
                        break
            
            # Save detailed drift report
            drift_details = {
                'drift_detected': drift_detected,
                'number_of_columns': n_features,
                'number_of_drifted_columns': n_drifted_features,
                'drift_ratio': n_drifted_features / n_features if n_features > 0 else 0,
                'timestamp': pd.Timestamp.now().isoformat(),
                'method': 'evidently'
            }
            
            # Save column-level drift details
            column_drifts = []
            for metric in report_result.get('metrics', []):
                if 'column_name' in metric.get('result', {}):
                    column_drifts.append({
                        'column': metric['result']['column_name'],
                        'drift_detected': metric['result'].get('drift_detected', False),
                        'drift_score': metric['result'].get('drift_score', 0)
                    })
            
            drift_details['column_drifts'] = column_drifts
            
            # Save report
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path, 
                content=drift_details
            )
            
            logging.info(f"📊 Drift detection completed: {n_drifted_features}/{n_features} features drifted")
            logging.info(f"🎯 Dataset drift detected: {drift_detected}")
            
            return drift_detected
            
        except Exception as e:
            logging.error(f"❌ Evidently drift detection failed: {e}")
            raise InsuranceException(e, sys)

    def _ensure_correct_data_types(self, df: DataFrame) -> DataFrame:
        """
        Ensure DataFrame has correct data types according to schema
        """
        try:
            df_copy = df.copy()
            
            # Convert numerical columns
            for col in self._schema_config.get("numerical_columns", []):
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            # Convert categorical columns  
            for col in self._schema_config.get("categorical_columns", []):
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].astype('category')
            
            return df_copy
            
        except Exception as e:
            logging.warning(f"Could not enforce data types: {e}")
            return df

    def _basic_statistical_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Basic statistical drift detection as fallback
        """
        try:
            logging.info("📊 Using basic statistical drift detection")
            
            numerical_cols = reference_df.select_dtypes(include=['number']).columns
            categorical_cols = reference_df.select_dtypes(include=['object', 'category']).columns
            
            drift_detected = False
            drift_details = {
                'numerical_drifts': {},
                'categorical_drifts': {},
                'method': 'basic_statistical'
            }
            
            # Check numerical columns (mean difference > 20%)
            for col in numerical_cols:
                if col in reference_df.columns and col in current_df.columns:
                    ref_mean = reference_df[col].mean()
                    curr_mean = current_df[col].mean()
                    
                    if ref_mean != 0:
                        mean_diff = abs(ref_mean - curr_mean) / ref_mean
                    else:
                        mean_diff = abs(ref_mean - curr_mean)
                    
                    drift_detected_col = mean_diff > 0.2  # 20% threshold
                    drift_details['numerical_drifts'][col] = {
                        'mean_difference': float(mean_diff),
                        'drift_detected': drift_detected_col,
                        'reference_mean': float(ref_mean),
                        'current_mean': float(curr_mean)
                    }
                    
                    if drift_detected_col:
                        drift_detected = True
                        logging.info(f"⚠️ Potential drift in {col}: mean difference {mean_diff:.2%}")
            
            # Check categorical columns (distribution difference)
            for col in categorical_cols:
                if col in reference_df.columns and col in current_df.columns:
                    ref_dist = reference_df[col].value_counts(normalize=True)
                    curr_dist = current_df[col].value_counts(normalize=True)
                    
                    # Align distributions
                    all_categories = set(ref_dist.index) | set(curr_dist.index)
                    ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
                    curr_aligned = curr_dist.reindex(all_categories, fill_value=0)
                    
                    # Calculate total variation distance
                    tv_distance = 0.5 * sum(abs(ref_aligned - curr_aligned))
                    drift_detected_col = tv_distance > 0.1  # 10% threshold
                    
                    drift_details['categorical_drifts'][col] = {
                        'tv_distance': float(tv_distance),
                        'drift_detected': drift_detected_col,
                        'reference_distribution': ref_dist.to_dict(),
                        'current_distribution': curr_dist.to_dict()
                    }
                    
                    if drift_detected_col:
                        drift_detected = True
                        logging.info(f"⚠️ Potential drift in {col}: TV distance {tv_distance:.2%}")
            
            # Save basic drift report
            drift_details['drift_detected'] = drift_detected
            drift_details['timestamp'] = pd.Timestamp.now().isoformat()
            
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path, 
                content=drift_details
            )
            
            logging.info(f"📊 Basic drift detection completed: {drift_detected}")
            return drift_detected
            
        except Exception as e:
            logging.error(f"❌ Basic statistical drift detection failed: {e}")
            # Save error report
            error_report = {
                'drift_detected': False,
                'error': str(e),
                'method': 'failed',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path, 
                content=error_report
            )
            return False

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            # Store validation results
            validation_results = {
                'train_shape': train_df.shape,
                'test_shape': test_df.shape,
                'train_columns': list(train_df.columns),
                'test_columns': list(test_df.columns),
                'train_missing_values': train_df.isnull().sum().to_dict(),
                'test_missing_values': test_df.isnull().sum().to_dict()
            }

            status = self.validate_number_of_columns(dataframe=train_df)
            validation_results['train_columns_valid'] = status
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            
            status = self.validate_number_of_columns(dataframe=test_df)
            validation_results['test_columns_valid'] = status
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)
            validation_results['train_columns_exist'] = status
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            
            status = self.is_column_exist(df=test_df)
            validation_results['test_columns_exist'] = status
            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                validation_results['drift_detected'] = drift_status
                if drift_status:
                    logging.info(f"⚠️ Drift detected between train and test datasets.")
                    validation_error_msg = "Data drift detected between train and test sets"
                else:
                    validation_error_msg = "No data drift detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            # Save validation results as artifact
            validation_results_path = os.path.join(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                'validation_results.yaml'
            )
            write_yaml_file(validation_results_path, validation_results)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def validate_data_types(self, dataframe: DataFrame) -> bool:
        """
        Validate data types against schema
        """
        try:
            schema_config = self._schema_config
            validation_errors = []
            
            for column_config in schema_config["columns"]:
                for col_name, expected_type in column_config.items():
                    if col_name in dataframe.columns:
                        actual_type = str(dataframe[col_name].dtype)
                        
                        # Map expected types to pandas dtypes
                        type_mapping = {
                            'int': 'int64',
                            'float': 'float64', 
                            'category': 'object'
                        }
                        
                        expected_pandas_type = type_mapping.get(expected_type, expected_type)
                        
                        if expected_pandas_type not in actual_type:
                            validation_errors.append(
                                f"Column {col_name}: expected {expected_type}, got {actual_type}"
                            )
            
            if validation_errors:
                logging.warning(f"Data type validation errors: {validation_errors}")
                return False
                
            return True
            
        except Exception as e:
            raise InsuranceException(e, sys)

    def perform_data_quality_check(self, dataframe: DataFrame, dataset_name: str) -> dict:
        """
        Perform comprehensive data quality check
        """
        try:
            # Simple data quality check without external dependencies
            quality_report = {
                'dataset': dataset_name,
                'shape': dataframe.shape,
                'missing_values': dataframe.isnull().sum().to_dict(),
                'duplicate_rows': dataframe.duplicated().sum(),
                'data_types': dataframe.dtypes.astype(str).to_dict(),
                'basic_stats': dataframe.describe().to_dict() if len(dataframe.select_dtypes(include=['number']).columns) > 0 else {}
            }
            
            # Save quality report
            quality_report_path = os.path.join(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                f'{dataset_name}_quality_report.yaml'
            )
            write_yaml_file(quality_report_path, quality_report)
            
            return quality_report
            
        except Exception as e:
            logging.warning(f"Data quality check failed: {e}")
            return {}