import os
import sys
import numpy as np
import pandas as pd
from insurance_charges.entity.config_entity import InsurancePredictorConfig
from insurance_charges.entity.s3_estimator import InsuranceEstimator
from insurance_charges.exception import InsuranceException
from insurance_charges.logger import logging
from insurance_charges.utils.main_utils import read_yaml_file
from pandas import DataFrame

class InsuranceData:
    def __init__(self,
                age: int,
                sex: str,
                bmi: float,
                children: int,
                smoker: str,
                region: str,
                company: str = "goutham"  # Changed from ineuron to goutham
                ):
        """
        Insurance Data constructor
        Input: all features of the trained model for prediction
        """
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
        """
        This function returns a DataFrame from InsuranceData class input
        """
        try:
            insurance_input_dict = self.get_insurance_data_as_dict()
            return DataFrame(insurance_input_dict)
        
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def get_insurance_data_as_dict(self):
        """
        This function returns a dictionary from InsuranceData class input 
        """
        logging.info("Entered get_insurance_data_as_dict method as InsuranceData class")

        try:
            input_data = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region]
            }

            logging.info("Created insurance data dict")
            logging.info("Exited get_insurance_data_as_dict method as InsuranceData class")

            return input_data

        except Exception as e:
            raise InsuranceException(e, sys) from e
    def validate_input_data(self, input_data: dict) -> Tuple[bool, str]:
        """
        Validate input data for prediction
        """
        try:
            # Age validation
            age = input_data.get('age', [0])[0]
            if not (18 <= age <= 100):
                return False, "Age must be between 18 and 100"
            
            # BMI validation
            bmi = input_data.get('bmi', [0])[0]
            if not (10 <= bmi <= 60):
                return False, "BMI must be between 10 and 60"
            
            # Children validation
            children = input_data.get('children', [0])[0]
            if not (0 <= children <= 20):
                return False, "Number of children must be between 0 and 20"
            
            # Categorical values validation
            valid_sex = ['male', 'female']
            sex = input_data.get('sex', [''])[0]
            if sex not in valid_sex:
                return False, f"Sex must be one of {valid_sex}"
            
            valid_smoker = ['yes', 'no']
            smoker = input_data.get('smoker', [''])[0]
            if smoker not in valid_smoker:
                return False, f"Smoker must be one of {valid_smoker}"
            
            valid_regions = ['southwest', 'southeast', 'northwest', 'northeast']
            region = input_data.get('region', [''])[0]
            if region not in valid_regions:
                return False, f"Region must be one of {valid_regions}"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class InsuranceClassifier:
    def __init__(self, prediction_pipeline_config: InsurancePredictorConfig = InsurancePredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def predict(self, dataframe) -> np.array:
        """
        This is the method of InsuranceClassifier
        Returns: Prediction in numpy array format (insurance charges)
        """
        try:
            logging.info("Entered predict method of InsuranceClassifier class")
            model = InsuranceEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            
            # Ensure we return a numpy array
            if isinstance(result, (list, pd.Series)):
                return np.array(result)
            return result
            
        except Exception as e:
            raise InsuranceException(e, sys)