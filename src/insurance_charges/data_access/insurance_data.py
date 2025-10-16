from insurance_charges.configuration.postgres_db_connection import PostgreSQLClient
from insurance_charges.constants import DATABASE_NAME
from insurance_charges.exception import InsuranceException
import pandas as pd
import sys
from typing import Optional

class InsuranceData:
    """
    This class helps to export entire PostgreSQL table as pandas dataframe
    """

    def __init__(self):
        """
        Initialize PostgreSQL client
        """
        try:
            # Using default connection string from environment
            self.postgres_client = PostgreSQLClient()
        except Exception as e:
            raise InsuranceException(e, sys)

    # In src/insurance_charges/data_access/insurance_data.py - UPDATE METHOD SIGNATURE:
    def export_table_as_dataframe(self, table_name: str = None, schema: str = None) -> pd.DataFrame:
        """
        Export data from existing PostgreSQL table
        """
        try:
            # Use provided table name or default from constants
            if table_name is None:
                from insurance_charges.constants import TABLE_NAME
                table_name = TABLE_NAME
            
            if schema:
                query = f"SELECT * FROM {schema}.{table_name}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            df = pd.read_sql_query(query, self.postgres_client.engine)
            
            logging.info(f"Successfully loaded {len(df)} records from table: {schema}.{table_name}" if schema else f"Successfully loaded {len(df)} records from table: {table_name}")
            return df
            
        except Exception as e:
            logging.error(f"Failed to load data from table {table_name}: {e}")
            raise InsuranceException(e, sys)