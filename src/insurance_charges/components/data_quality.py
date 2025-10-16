import pandas as pd
import numpy as np
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
import sys

class DataQualityChecker:
    """
    Comprehensive data quality checks
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.quality_report = {}
    
    def check_missing_values(self):
        """Check for missing values"""
        missing_values = self.dataframe.isnull().sum()
        missing_percentage = (missing_values / len(self.dataframe)) * 100
        
        self.quality_report['missing_values'] = {
            'counts': missing_values.to_dict(),
            'percentages': missing_percentage.to_dict()
        }
        
        high_missing = missing_percentage[missing_percentage > 5]
        if not high_missing.empty:
            logging.warning(f"High missing values detected: {high_missing.to_dict()}")
    
    def check_data_types(self):
        """Check data types consistency"""
        dtypes = self.dataframe.dtypes.astype(str).to_dict()
        self.quality_report['data_types'] = dtypes
    
    def check_duplicates(self):
        """Check for duplicate rows"""
        duplicate_count = self.dataframe.duplicated().sum()
        self.quality_report['duplicates'] = {
            'count': int(duplicate_count),
            'percentage': float((duplicate_count / len(self.dataframe)) * 100)
        }
    
    def check_value_ranges(self):
        """Check for unrealistic value ranges"""
        value_ranges = {}
        
        # Age validation
        if 'age' in self.dataframe.columns:
            invalid_age = self.dataframe[(self.dataframe['age'] < 18) | (self.dataframe['age'] > 100)]
            value_ranges['age'] = {
                'invalid_count': len(invalid_age),
                'min_allowed': 18,
                'max_allowed': 100
            }
        
        # BMI validation
        if 'bmi' in self.dataframe.columns:
            invalid_bmi = self.dataframe[(self.dataframe['bmi'] < 10) | (self.dataframe['bmi'] > 60)]
            value_ranges['bmi'] = {
                'invalid_count': len(invalid_bmi),
                'min_allowed': 10,
                'max_allowed': 60
            }
        
        self.quality_report['value_ranges'] = value_ranges
    
    def check_categorical_values(self):
        """Validate categorical values"""
        categorical_checks = {}
        
        expected_categories = {
            'sex': ['male', 'female'],
            'smoker': ['yes', 'no'],
            'region': ['southwest', 'southeast', 'northwest', 'northeast']
        }
        
        for col, expected_values in expected_categories.items():
            if col in self.dataframe.columns:
                unique_values = self.dataframe[col].unique().tolist()
                unexpected_values = [val for val in unique_values if val not in expected_values]
                
                categorical_checks[col] = {
                    'expected_values': expected_values,
                    'actual_values': unique_values,
                    'unexpected_values': unexpected_values
                }
        
        self.quality_report['categorical_values'] = categorical_checks
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        try:
            self.check_missing_values()
            self.check_data_types()
            self.check_duplicates()
            self.check_value_ranges()
            self.check_categorical_values()
            
            # Overall quality score
            quality_score = 100
            penalties = 0
            
            # Penalize for issues
            if self.quality_report['missing_values']['counts']:
                total_missing = sum(self.quality_report['missing_values']['counts'].values())
                penalties += min(total_missing / len(self.dataframe) * 100, 30)
            
            if self.quality_report['duplicates']['count'] > 0:
                penalties += min(self.quality_report['duplicates']['percentage'], 20)
            
            quality_score -= penalties
            self.quality_report['quality_score'] = max(0, quality_score)
            
            logging.info(f"Data quality score: {quality_score:.2f}")
            return self.quality_report
            
        except Exception as e:
            raise InsuranceException(e, sys)