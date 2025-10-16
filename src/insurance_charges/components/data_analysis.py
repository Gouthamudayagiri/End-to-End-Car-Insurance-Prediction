import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from src.insurance_charges.exception import InsuranceException
from src.insurance_charges.logger import logging
from src.insurance_charges.utils.main_utils import write_yaml_file

class DataAnalysis:
    """
    Perform comprehensive data analysis similar to the notebook and save artifacts
    """
    
    def __init__(self, data_ingestion_artifact, artifact_dir: str):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
    
    def load_data(self):
        """Load training and test data"""
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def detect_outliers(self, df, features):
        """Outlier detection from notebook"""
        try:
            outlier_indices = []
            for col in features:
                Q1 = np.percentile(df[col], 25)
                Q3 = np.percentile(df[col], 75)
                IQR = Q3 - Q1
                outlier_step = 1.5 * IQR
                outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
                outlier_indices.extend(outlier_list_col)
            outlier_indices = list(set(outlier_indices))
            return outlier_indices
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def handle_outliers(self, df):
        """Handle outliers using winsorization from notebook"""
        try:
            df_copy = df.copy()
            numerical_cols = ['bmi', 'charges']
            
            for col in numerical_cols:
                if col in df_copy.columns:
                    df_copy[col] = winsorize(df_copy[col], limits=[0.05, 0.05])
            
            return df_copy
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def generate_correlation_matrix(self, df, save_path=None):
        """Generate correlation matrix from notebook"""
        try:
            corr_matrix = df.corr(numeric_only=True)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm", center=0)
            plt.title("Correlation Matrix")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Correlation matrix saved to: {save_path}")
            
            plt.close()
            return corr_matrix
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def generate_distribution_plots(self, df, save_dir):
        """Generate distribution plots for numerical features"""
        try:
            numerical_features = ['age', 'bmi', 'children', 'charges']
            
            for feature in numerical_features:
                if feature in df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 2, 1)
                    sns.histplot(df[feature], kde=True)
                    plt.title(f'Distribution of {feature}')
                    
                    plt.subplot(1, 2, 2)
                    sns.boxplot(y=df[feature])
                    plt.title(f'Boxplot of {feature}')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'{feature}_distribution.png'), dpi=300)
                    plt.close()
                    
        except Exception as e:
            logging.warning(f"Could not generate distribution plots: {e}")
    
    def perform_analysis(self):
        """Perform complete data analysis and save artifacts"""
        try:
            train_df, test_df = self.load_data()
            
            # Create analysis directory
            plots_dir = os.path.join(self.artifact_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Handle outliers
            train_df_clean = self.handle_outliers(train_df)
            test_df_clean = self.handle_outliers(test_df)
            
            # Detect outliers
            numerical_features = ['age', 'bmi', 'children', 'charges']
            train_outliers = self.detect_outliers(train_df, numerical_features)
            test_outliers = self.detect_outliers(test_df, numerical_features)
            
            logging.info(f"Training outliers: {len(train_outliers)}")
            logging.info(f"Test outliers: {len(test_outliers)}")
            
            # Generate plots
            self.generate_distribution_plots(train_df_clean, plots_dir)
            
            # Generate correlation matrix
            corr_matrix_path = os.path.join(plots_dir, 'correlation_matrix.png')
            corr_matrix = self.generate_correlation_matrix(train_df_clean, corr_matrix_path)
            
            # Generate descriptive statistics
            descriptive_stats = {
                'train_descriptive': train_df.describe().to_dict(),
                'test_descriptive': test_df.describe().to_dict(),
                'categorical_summary': {
                    'sex': train_df['sex'].value_counts().to_dict(),
                    'smoker': train_df['smoker'].value_counts().to_dict(),
                    'region': train_df['region'].value_counts().to_dict()
                }
            }
            
            analysis_report = {
                'original_train_shape': train_df.shape,
                'original_test_shape': test_df.shape,
                'cleaned_train_shape': train_df_clean.shape,
                'cleaned_test_shape': test_df_clean.shape,
                'train_outliers_count': len(train_outliers),
                'test_outliers_count': len(test_outliers),
                'correlation_with_target': corr_matrix['charges'].sort_values(ascending=False).to_dict(),
                'data_quality': {
                    'train_missing_values': train_df.isnull().sum().to_dict(),
                    'test_missing_values': test_df.isnull().sum().to_dict(),
                    'train_duplicates': train_df.duplicated().sum(),
                    'test_duplicates': test_df.duplicated().sum()
                }
            }
            
            # Merge with descriptive stats
            analysis_report.update(descriptive_stats)
            
            # Save analysis report as artifact
            analysis_report_path = os.path.join(self.artifact_dir, 'analysis_report.yaml')
            write_yaml_file(analysis_report_path, analysis_report)
            
            # Save cleaned data
            train_clean_path = os.path.join(self.artifact_dir, 'train_cleaned.csv')
            test_clean_path = os.path.join(self.artifact_dir, 'test_cleaned.csv')
            train_df_clean.to_csv(train_clean_path, index=False)
            test_df_clean.to_csv(test_clean_path, index=False)
            
            logging.info(f"Data analysis artifacts saved to: {self.artifact_dir}")
            
            return analysis_report
            
        except Exception as e:
            raise InsuranceException(e, sys)