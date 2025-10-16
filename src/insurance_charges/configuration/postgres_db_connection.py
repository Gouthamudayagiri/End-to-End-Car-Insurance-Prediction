import sys
import os
import pandas as pd
from sqlalchemy import create_engine
from insurance_charges.exception import InsuranceException
from insurance_charges.logger import logging
from insurance_charges.constants import POSTGRES_URL_KEY

class PostgreSQLClient:
    """
    Class to handle PostgreSQL database connections and operations
    """
    client = None

    def __init__(self, database_url: str = None) -> None:
        try:
            if database_url is None:
                database_url = os.getenv(POSTGRES_URL_KEY)
                if database_url is None:
                    raise Exception(f"Environment key: {POSTGRES_URL_KEY} is not set.")
            
            self.database_url = database_url
            self.engine = create_engine(database_url)
            logging.info("PostgreSQL connection successful")
        except Exception as e:
            raise InsuranceException(e, sys)

    def export_table_as_dataframe(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """
        Export entire table as dataframe
        """
        try:
            if schema:
                query = f"SELECT * FROM {schema}.{table_name}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            df = pd.read_sql_query(query, self.engine)
            logging.info(f"Exported {len(df)} records from {table_name}")
            return df
        except Exception as e:
            raise InsuranceException(e, sys)

    def save_dataframe_to_table(self, dataframe: pd.DataFrame, table_name: str, 
                               schema: str = None, if_exists: str = 'replace'):
        """
        Save dataframe to PostgreSQL table
        """
        try:
            if schema:
                table_name = f"{schema}.{table_name}"
            
            dataframe.to_sql(
                table_name, 
                self.engine, 
                if_exists=if_exists, 
                index=False
            )
            logging.info(f"Saved {len(dataframe)} records to {table_name}")
        except Exception as e:
            raise InsuranceException(e, sys)