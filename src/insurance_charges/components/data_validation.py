import json
import sys
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount
from pandas import DataFrame

from insurance_charges.exception import InsuranceException
from insurance_charges.logger import logging
from insurance_charges.utils.main_utils import read_yaml_file, write_yaml_file
from insurance_charges.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from insurance_charges.entity.config_entity import DataValidationConfig
from insurance_charges.constants import SCHEMA_FILE_PATH

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
            drift_metric = DriftedColumnsCount()
            data_drift_report = Report(metrics=[drift_metric])
            
            data_drift_report.run(reference_data=reference_df, current_data=current_df)
            
            metric_dict = drift_metric.dict()
            
            dataset_drift = False
            n_features = 0
            n_drifted_features = 0
            
            if 'result' in metric_dict:
                result = metric_dict['result']
                n_features = result.get('number_of_columns', 0)
                n_drifted_features = result.get('number_of_drifted_columns', 0)
                dataset_drift = n_drifted_features > 0
            
            try:
                report_dict = {
                    'drift_detected': dataset_drift,
                    'number_of_columns': n_features,
                    'number_of_drifted_columns': n_drifted_features,
                    'metric_details': metric_dict
                }
                write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report_dict)
            except Exception as save_error:
                logging.info(f"Could not save drift report to file: {save_error}")
            
            logging.info(f"Dataset drift detection completed: {n_drifted_features}/{n_features} features drifted.")
            
            return dataset_drift
        except Exception as e:
            raise InsuranceException(e, sys) from e

    # def initiate_data_validation(self) -> DataValidationArtifact:
    #     """
    #     Method Name :   initiate_data_validation
    #     Description :   This method initiates the data validation component for the pipeline
    #     """
    #     try:
    #         validation_error_msg = ""
    #         logging.info("Starting data validation")
            
    #         train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
    #                              DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

    #         status = self.validate_number_of_columns(dataframe=train_df)
    #         logging.info(f"All required columns present in training dataframe: {status}")
    #         if not status:
    #             validation_error_msg += f"Columns are missing in training dataframe."
            
    #         status = self.validate_number_of_columns(dataframe=test_df)
    #         logging.info(f"All required columns present in testing dataframe: {status}")
    #         if not status:
    #             validation_error_msg += f"Columns are missing in test dataframe."

    #         status = self.is_column_exist(df=train_df)
    #         if not status:
    #             validation_error_msg += f"Columns are missing in training dataframe."
            
    #         status = self.is_column_exist(df=test_df)
    #         if not status:
    #             validation_error_msg += f"columns are missing in test dataframe."

    #         validation_status = len(validation_error_msg) == 0

    #         if validation_status:
    #             drift_status = self.detect_dataset_drift(train_df, test_df)
    #             if drift_status:
    #                 logging.info(f"Drift detected.")
    #                 validation_error_msg = "Drift detected"
    #             else:
    #                 validation_error_msg = "Drift not detected"
    #         else:
    #             logging.info(f"Validation_error: {validation_error_msg}")

    #         data_validation_artifact = DataValidationArtifact(
    #             validation_status=validation_status,
    #             message=validation_error_msg,
    #             drift_report_file_path=self.data_validation_config.drift_report_file_path
    #         )

    #         logging.info(f"Data validation artifact: {data_validation_artifact}")
    #         return data_validation_artifact
    #     except Exception as e:
    #         raise InsuranceException(e, sys) from 
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
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            # Save validation results as artifact
            validation_results_path = os.path.join(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                'validation_results.yaml'
            )
            from insurance_charges.utils.main_utils import write_yaml_file
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
            from insurance_charges.components.data_quality import DataQualityChecker
            
            quality_checker = DataQualityChecker(dataframe)
            quality_report = quality_checker.generate_quality_report()
            
            # Save quality report
            quality_report_path = os.path.join(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                f'{dataset_name}_quality_report.yaml'
            )
            from insurance_charges.utils.main_utils import write_yaml_file
            write_yaml_file(quality_report_path, quality_report)
            
            return quality_report
            
        except Exception as e:
            logging.warning(f"Data quality check failed: {e}")
            return {}    