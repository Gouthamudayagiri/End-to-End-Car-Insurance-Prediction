import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from typing import Tuple

from insurance_charges.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from insurance_charges.entity.config_entity import DataTransformationConfig
from insurance_charges.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from insurance_charges.exception import InsuranceException
from insurance_charges.logger import logging
from insurance_charges.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: Output reference of data validation artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsuranceException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise InsuranceException(e, sys)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features based on business logic"""
        try:
            df_copy = df.copy()
            
            # Create risk score feature (from your notebook analysis)
            df_copy['risk_score'] = (df_copy['age'] / 10) * (df_copy['bmi'] / 25) * df_copy["smoker"].map({"yes": 3, "no": 1})
            
            # Create BMI categories
            df_copy['bmi_category'] = pd.cut(
                df_copy['bmi'], 
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['underweight', 'normal', 'overweight', 'obese1', 'obese2']
            )
            
            # Children flag
            df_copy['children_flag'] = df_copy['children'].apply(lambda x: 1 if x > 0 else 0)
            
            logging.info("Successfully created new features")
            return df_copy
            
        except Exception as e:
            raise InsuranceException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Get column lists from schema config
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initializing preprocessing transformers")

            # Create transformers
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
            ordinal_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')  # Using OneHot for ordinal too
            
            # Power transformer for skewed features
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ],
                remainder='passthrough',
                n_jobs=-1
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            
            return preprocessor

        except Exception as e:
            raise InsuranceException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(f"Data validation failed: {self.data_validation_artifact.message}")

            logging.info("Starting data transformation")
            
            # Get preprocessor
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            # Read data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Testing data shape: {test_df.shape}")

            # Feature engineering
            train_df = self._create_features(train_df)
            test_df = self._create_features(test_df)

            # Prepare features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Successfully applied preprocessing transformations")

            # Create final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved all data transformation artifacts")

            # Create and return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            logging.error("Error during data transformation")
            raise InsuranceException(e, sys) from e
    def _prepare_target_variable(self, target_series: pd.Series) -> pd.Series:
        """
        Prepare target variable - handle any transformations needed
        """
        try:
            # Apply log transformation if specified in schema
            if self._schema_config.get('apply_log_to_target', False):
                return np.log1p(target_series)
            return target_series
        except Exception as e:
            raise InsuranceException(e, sys)

    def _inverse_transform_target(self, transformed_target: np.array) -> np.array:
        """
        Inverse transform target variable for predictions
        """
        try:
            if self._schema_config.get('apply_log_to_target', False):
                return np.expm1(transformed_target)
            return transformed_target
        except Exception as e:
            raise InsuranceException(e, sys)