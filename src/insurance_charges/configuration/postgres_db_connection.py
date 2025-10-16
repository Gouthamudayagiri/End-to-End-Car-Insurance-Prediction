# src/insurance_charges/configuration/postgres_db_connection.py
import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from src.insurance_charges.constants import POSTGRES_URL_KEY

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
                    # Use a more direct exception without complex error_detail
                    raise Exception(f"Environment variable {POSTGRES_URL_KEY} is not set. Please check your .env file.")
            
            # Clean the URL (remove any quotes or extra spaces)
            database_url = str(database_url).strip().strip('"').strip("'")
            
            logging.info(f"Attempting to connect to PostgreSQL...")
            
            # Direct connection attempt without complex validation
            self.database_url = database_url
            self.engine = create_engine(database_url, pool_pre_ping=True, echo=False)
            
            # Test connection
            self._test_connection()
            
            logging.info("✅ PostgreSQL connection successful")
            
        except Exception as e:
            logging.error(f"❌ Failed to connect to PostgreSQL: {str(e)}")
            # Use simple exception to avoid the traceback issue
            raise Exception(f"PostgreSQL connection failed: {str(e)}")

    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logging.info("✅ Database connection test passed")
                
        except SQLAlchemyError as e:
            raise Exception(f"Database connection test failed: {str(e)}")

    def export_table_as_dataframe(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """
        Export entire table as dataframe
        """
        try:
            if schema:
                full_table_name = f"{schema}.{table_name}"
                query = f"SELECT * FROM {full_table_name}"
            else:
                full_table_name = table_name
                query = f"SELECT * FROM {table_name}"
            
            logging.info(f"Executing query: {query}")
            
            df = pd.read_sql_query(query, self.engine)
            logging.info(f"✅ Exported {len(df)} records from {full_table_name}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to export table {table_name}: {str(e)}")
            raise Exception(f"Failed to export table {table_name}: {str(e)}")

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
                index=False,
                method='multi'
            )
            logging.info(f"✅ Saved {len(dataframe)} records to {table_name}")
        except Exception as e:
            raise Exception(f"Failed to save dataframe: {str(e)}")
    def get_table_names(self) -> list:
        """Get list of all tables in the database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                return [row[0] for row in result]
        except Exception as e:
            raise InsuranceException(e, sys)