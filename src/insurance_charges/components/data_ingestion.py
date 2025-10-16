# src/insurance_charges/components/data_ingestion.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.insurance_charges.entity.config_entity import DataIngestionConfig
from src.insurance_charges.entity.artifact_entity import DataIngestionArtifact
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from src.insurance_charges.data_access.insurance_data import InsuranceData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from PostgreSQL to csv file
        """
        try:
            logging.info(f"Exporting data from PostgreSQL")
            insurance_data = InsuranceData()
            dataframe = insurance_data.export_table_as_dataframe(
                table_name=self.data_ingestion_config.table_name,
                schema=self.data_ingestion_config.schema_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise InsuranceException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            logging.info("Performed train test split on the dataframe")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from PostgreSQL")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys) from e