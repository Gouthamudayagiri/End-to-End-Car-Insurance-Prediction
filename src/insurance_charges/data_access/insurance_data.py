# src/insurance_charges/data_access/insurance_data.py
from src.insurance_charges.configuration.postgres_db_connection import PostgreSQLClient
from src.insurance_charges.constants import TABLE_NAME
from src.insurance_charges.logger import logging
import pandas as pd
import sys

class InsuranceData:
    """
    This class helps to export entire PostgreSQL table as pandas dataframe
    """

    def __init__(self):
        """
        Initialize PostgreSQL client
        """
        try:
            self.postgres_client = PostgreSQLClient()
        except Exception as e:
            logging.error(f"Failed to initialize PostgreSQL client: {str(e)}")
            raise Exception(f"Database initialization failed: {str(e)}")

    def export_table_as_dataframe(self, table_name: str = None, schema: str = None) -> pd.DataFrame:
        """
        Export data from existing PostgreSQL table
        """
        try:
            # Use provided table name or default from constants
            if table_name is None:
                table_name = TABLE_NAME
            
            logging.info(f"Loading data from table: {table_name}")
            
            df = self.postgres_client.export_table_as_dataframe(
                table_name=table_name,
                schema=schema
            )
            
            logging.info(f"✅ Successfully loaded {len(df)} records from table: {table_name}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to load data from table {table_name}: {str(e)}")
            raise Exception(f"Data loading failed: {str(e)}")