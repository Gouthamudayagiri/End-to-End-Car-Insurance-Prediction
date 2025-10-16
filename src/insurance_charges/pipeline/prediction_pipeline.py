import os
import sys
import numpy as np
import pandas as pd
from src.insurance_charges.entity.config_entity import InsurancePredictorConfig
from src.insurance_charges.entity.s3_estimator import InsuranceEstimator
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from pandas import DataFrame

class InsuranceData:
    def __init__(self,
                age: int,
                sex: str,
                bmi: float,
                children: int,
                smoker: str,
                region: str,
                company: str = "goutham"
                ):
        try:
            self.age = age
            self.sex = sex
            self.bmi = bmi
            self.children = children
            self.smoker = smoker
            self.region = region
            self.company = company

        except Exception as e:
            raise InsuranceException(e, sys) from e

    def get_insurance_input_data_frame(self) -> DataFrame:
        try:
            # Calculate engineered features
            risk_score = (self.age * 0.05) + (self.bmi * 0.1)
            children_flag = 1 if self.children > 0 else 0
            smoker_numeric = 1 if self.smoker == 'yes' else 0
            smoker_bmi_interaction = smoker_numeric * self.bmi
            
            # Age category as NUMERICAL (preprocessor expects numerical)
            if self.age <= 25:
                age_category = 0  # young
            elif self.age <= 35:
                age_category = 1  # adult  
            elif self.age <= 45:
                age_category = 2  # middle_aged
            elif self.age <= 55:
                age_category = 3  # senior
            else:
                age_category = 4  # elderly
            
            # Create DataFrame with EXACT types preprocessor expects
            input_data = {
                # NUMERICAL columns (will be scaled by StandardScaler)
                'age': [self.age],                           
                'bmi': [self.bmi],                           
                'children': [self.children],                 
                'risk_score': [risk_score],                  
                'children_flag': [children_flag],            
                'smoker_bmi_interaction': [smoker_bmi_interaction],
                'age_category': [age_category],              # NUMERICAL!
                
                # CATEGORICAL columns (will be encoded by OneHotEncoder)
                'sex': [self.sex],                           # categorical string
                'smoker': [self.smoker],                     # categorical string  
                'region': [self.region]                      # categorical string
            }
            
            input_df = DataFrame(input_data)
            
            # Ensure correct column order (important for some models)
            expected_columns = [
                'age', 'bmi', 'children', 'risk_score', 'children_flag', 
                'smoker_bmi_interaction', 'age_category', 'sex', 'smoker', 'region'
            ]
            input_df = input_df[expected_columns]
            
            logging.info(f"✅ Created DataFrame with correct preprocessor expectations")
            logging.info(f"Numerical columns: {['age', 'bmi', 'children', 'risk_score', 'children_flag', 'smoker_bmi_interaction', 'age_category']}")
            logging.info(f"Categorical columns: {['sex', 'smoker', 'region']}")
            logging.info(f"Final DataFrame:\n{input_df}")
            
            return input_df
        
        except Exception as e:
            logging.error(f"Error creating input DataFrame: {e}")
            raise InsuranceException(e, sys) from e

class InsuranceClassifier:
    def __init__(self, prediction_pipeline_config: InsurancePredictorConfig = InsurancePredictorConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def predict(self, dataframe) -> np.array:
        try:
            logging.info("Entered predict method of InsuranceClassifier class")
            model = InsuranceEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            
            logging.info(f"DataFrame for prediction:")
            logging.info(f"Shape: {dataframe.shape}")
            logging.info(f"Columns: {dataframe.columns.tolist()}")
            logging.info(f"Dtypes: {dataframe.dtypes.tolist()}")
            
            result = model.predict(dataframe)
            
            if isinstance(result, (list, pd.Series)):
                result = np.array(result)
            
            logging.info(f"🎯 Prediction successful: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise InsuranceException(e, sys)