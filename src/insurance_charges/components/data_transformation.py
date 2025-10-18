# src/insurance_charges/components/data_transformation.py
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple
from scipy import sparse

from src.insurance_charges.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.insurance_charges.entity.config_entity import DataTransformationConfig
from src.insurance_charges.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from src.insurance_charges.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns

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
        """Create new features based on business logic - only numerical features"""
        try:
            df_copy = df.copy()
            
            # Create risk score feature (numerical)
            smoker_map = {"yes": 3, "no": 1}
            df_copy['risk_score'] = (df_copy['age'] / 10) * (df_copy['bmi'] / 25) * df_copy["smoker"].map(smoker_map)
            
            # Children flag (numerical)
            df_copy['children_flag'] = df_copy['children'].apply(lambda x: 1 if x > 0 else 0)
            
            # Smoker-BMI interaction (numerical)
            df_copy['smoker_bmi_interaction'] = df_copy['bmi'] * df_copy["smoker"].map(smoker_map)
            
            # Age categories (numerical encoding instead of categorical)
            age_bins = [0, 30, 40, 50, 60, 100]
            age_labels = [1, 2, 3, 4, 5]
            df_copy['age_category'] = pd.cut(
                df_copy['age'], 
                bins=age_bins,
                labels=age_labels
            ).astype(int)
            
            # BMI categories (numerical)
            bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
            bmi_labels = [1, 2, 3, 4, 5, 6]
            df_copy['bmi_category'] = pd.cut(
                df_copy['bmi'],
                bins=bmi_bins,
                labels=bmi_labels
            ).astype(int)
            
            logging.info(f"✅ Created new features. DataFrame shape: {df_copy.shape}")
            logging.info(f"New columns: {[col for col in df_copy.columns if col not in df.columns]}")
            
            return df_copy
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise InsuranceException(e, sys)

    def get_data_transformer_object(self, feature_columns: list) -> ColumnTransformer:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Define numerical and categorical columns explicitly
            base_numerical = ['age', 'bmi', 'children']
            engineered_numerical = ['risk_score', 'children_flag', 'smoker_bmi_interaction', 'age_category', 'bmi_category']
            categorical_columns = ['sex', 'smoker', 'region']
            
            # Combine base and engineered numerical features
            all_numerical_columns = base_numerical + engineered_numerical
            
            # Filter to only include columns that actually exist in the data
            numerical_columns = [col for col in all_numerical_columns if col in feature_columns]
            categorical_columns = [col for col in categorical_columns if col in feature_columns]
            
            logging.info(f"Final Numerical columns ({len(numerical_columns)}): {numerical_columns}")
            logging.info(f"Final Categorical columns ({len(categorical_columns)}): {categorical_columns}")
            
            # Check for missing columns
            missing_columns = set(feature_columns) - set(numerical_columns + categorical_columns)
            if missing_columns:
                logging.warning(f"Columns not processed: {missing_columns}")

            # Create transformers
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            
            # Create transformers list
            transformers = []
            
            if numerical_columns:
                transformers.append(("Numerical", numeric_transformer, numerical_columns))
                logging.info(f"Added numerical transformer for {len(numerical_columns)} columns")
            
            if categorical_columns:
                transformers.append(("Categorical", categorical_transformer, categorical_columns))
                logging.info(f"Added categorical transformer for {len(categorical_columns)} columns")

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop',  # Drop columns not specified
                n_jobs=-1
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            
            return preprocessor

        except Exception as e:
            logging.error(f"Error creating data transformer: {e}")
            raise InsuranceException(e, sys) from e

    def _ensure_dense_array(self, array):
        """Convert sparse matrix to dense array if needed"""
        if sparse.issparse(array):
            logging.info("Converting sparse matrix to dense array")
            return array.toarray()
        return array

    def _ensure_proper_dtype(self, array):
        """Ensure array has proper dtype for numpy saving"""
        try:
            # Convert to numpy array with proper dtype
            if array.dtype == object:
                array = np.array(array, dtype=np.float64)
            return array
        except Exception as e:
            logging.warning(f"Could not convert to float64: {e}, using object dtype")
            return np.array(array, dtype=object)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(f"Data validation failed: {self.data_validation_artifact.message}")

            logging.info("Starting data transformation")
            
            # Read data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logging.info(f"Original Training data shape: {train_df.shape}")
            logging.info(f"Original Testing data shape: {test_df.shape}")
            logging.info(f"Training columns: {list(train_df.columns)}")

            # Feature engineering - create new features
            train_df = self._create_features(train_df)
            test_df = self._create_features(test_df)

            logging.info(f"After feature engineering - Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            logging.info(f"Train columns after feature engineering: {list(train_df.columns)}")

            # Prepare features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"Input features columns: {list(input_feature_train_df.columns)}")
            logging.info(f"Target column: {TARGET_COLUMN}")

            # Get preprocessor with actual feature columns
            feature_columns = list(input_feature_train_df.columns)
            preprocessor = self.get_data_transformer_object(feature_columns=feature_columns)

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Ensure arrays are dense and have proper dtypes
            input_feature_train_arr = self._ensure_dense_array(input_feature_train_arr)
            input_feature_test_arr = self._ensure_dense_array(input_feature_test_arr)
            
            input_feature_train_arr = self._ensure_proper_dtype(input_feature_train_arr)
            input_feature_test_arr = self._ensure_proper_dtype(input_feature_test_arr)

            logging.info("Successfully applied preprocessing transformations")
            logging.info(f"Transformed train shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed test shape: {input_feature_test_arr.shape}")
            logging.info(f"Transformed train dtype: {input_feature_train_arr.dtype}")
            logging.info(f"Transformed test dtype: {input_feature_test_arr.dtype}")

            # Prepare target arrays with proper dtype
            target_feature_train_arr = self._ensure_proper_dtype(np.array(target_feature_train_df))
            target_feature_test_arr = self._ensure_proper_dtype(np.array(target_feature_test_df))

            # Create final arrays
            train_arr = np.column_stack([input_feature_train_arr, target_feature_train_arr])
            test_arr = np.column_stack([input_feature_test_arr, target_feature_test_arr])

            logging.info(f"Final train array shape: {train_arr.shape}, dtype: {train_arr.dtype}")
            logging.info(f"Final test array shape: {test_arr.shape}, dtype: {test_arr.dtype}")

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
            logging.error(f"Error during data transformation: {e}")
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