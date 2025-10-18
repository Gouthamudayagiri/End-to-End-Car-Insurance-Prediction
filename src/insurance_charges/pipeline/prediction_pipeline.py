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
            # Calculate engineered features EXACTLY as in training
            smoker_map = {"yes": 3, "no": 1}
            risk_score = (self.age / 10) * (self.bmi / 25) * smoker_map[self.smoker]
            children_flag = 1 if self.children > 0 else 0
            smoker_bmi_interaction = self.bmi * smoker_map[self.smoker]
            
            # Age category as NUMERICAL (must match training)
            if self.age <= 30:
                age_category = 1
            elif self.age <= 40:
                age_category = 2
            elif self.age <= 50:
                age_category = 3
            elif self.age <= 60:
                age_category = 4
            else:
                age_category = 5
            
            # BMI category as NUMERICAL (must match training)
            if self.bmi <= 18.5:
                bmi_category = 1
            elif self.bmi <= 25:
                bmi_category = 2
            elif self.bmi <= 30:
                bmi_category = 3
            elif self.bmi <= 35:
                bmi_category = 4
            elif self.bmi <= 40:
                bmi_category = 5
            else:
                bmi_category = 6
            
            # Create DataFrame with EXACT types and order as preprocessor expects
            input_data = {
                # Base numerical columns
                'age': [self.age],                           
                'bmi': [self.bmi],                           
                'children': [self.children],                 
                
                # Engineered numerical features
                'risk_score': [risk_score],                  
                'children_flag': [children_flag],            
                'smoker_bmi_interaction': [smoker_bmi_interaction],
                'age_category': [age_category],              
                'bmi_category': [bmi_category],
                
                # Categorical columns  
                'sex': [self.sex],                           
                'smoker': [self.smoker],                     
                'region': [self.region]                      
            }
            
            input_df = DataFrame(input_data)
            
            # Ensure correct column order (CRITICAL for preprocessor)
            expected_columns = [
                'age', 'bmi', 'children', 'risk_score', 'children_flag', 
                'smoker_bmi_interaction', 'age_category', 'bmi_category',
                'sex', 'smoker', 'region'
            ]
            
            # Reorder columns to match training data
            input_df = input_df[expected_columns]
            
            logging.info(f"✅ Created DataFrame with {len(input_df.columns)} features")
            logging.info(f"Numerical features: {expected_columns[:8]}")
            logging.info(f"Categorical features: {expected_columns[8:]}")
            logging.info(f"DataFrame shape: {input_df.shape}")
            
            return input_df
        
        except Exception as e:
            logging.error(f"❌ Error creating input DataFrame: {e}")
            raise InsuranceException(e, sys) from e

class InsuranceClassifier:
    def __init__(self, prediction_pipeline_config: InsurancePredictorConfig = InsurancePredictorConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model = None
            self._load_model()
        except Exception as e:
            raise InsuranceException(e, sys)

    def _load_model(self):
        """Load the model on initialization"""
        try:
            self.model = InsuranceEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            logging.info("✅ Model loaded successfully")
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            raise InsuranceException(e, sys)

    def predict(self, dataframe) -> np.array:
        try:
            logging.info("Entered predict method of InsuranceClassifier class")
            
            if self.model is None:
                self._load_model()
            
            logging.info(f"📊 Input DataFrame for prediction:")
            logging.info(f"Shape: {dataframe.shape}")
            logging.info(f"Columns: {dataframe.columns.tolist()}")
            logging.info(f"Dtypes: {dataframe.dtypes.tolist()}")
            logging.info(f"First row values: {dataframe.iloc[0].tolist()}")
            
            result = self.model.predict(dataframe)
            
            if isinstance(result, (list, pd.Series)):
                result = np.array(result)
            
            logging.info(f"🎯 Prediction successful: ${float(result[0]):,.2f}")
            return result
            
        except Exception as e:
            logging.error(f"❌ Prediction failed: {e}")
            raise InsuranceException(e, sys)