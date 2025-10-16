import sys
import pandas as pd
from pandas import DataFrame  # ADD THIS IMPORT
from sklearn.pipeline import Pipeline
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
import datetime
import json

class InsuranceModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.model_version = "1.0.0"
        self.created_date = datetime.datetime.now().isoformat()
        self.model_type = type(trained_model_object).__name__

    def get_model_info(self) -> dict:
        """
        Return model metadata
        """
        return {
            "model_version": self.model_version,
            "created_date": self.created_date,
            "model_type": self.model_type,
            "preprocessor_type": type(self.preprocessing_object).__name__
        }

    def predict(self, dataframe: DataFrame) -> pd.DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of InsuranceModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise InsuranceException(e, sys) from e

    def __repr__(self):
        return f"{self.model_type}(version={self.model_version})"

    def __str__(self):
        return f"{self.model_type}(version={self.model_version})"