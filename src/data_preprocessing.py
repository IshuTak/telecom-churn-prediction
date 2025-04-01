import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

class TelecomDataPreprocessor:
    def __init__(self, file_path: str):
        
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path
        self.raw_data = None
    
    def load_data(self) -> pd.DataFrame:
        
        try:
            self.raw_data = pd.read_csv(self.file_path)
            self.logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        
        if self.raw_data is None:
            self.load_data()
        
        
        df_cleaned = self.raw_data.copy()
        
        
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        df_cleaned.dropna(inplace=True)
        
        
        binary_columns = [
            'Churn', 'gender', 'Partner', 'Dependents', 
            'PhoneService', 'PaperlessBilling', 'SeniorCitizen'
        ]
        
        for col in binary_columns:
            if df_cleaned[col].dtype == 'category':
                df_cleaned[col] = df_cleaned[col].map({'Yes': 1, 'No': 0})
            elif df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].map({'Yes': 1, 'No': 0})
        
        
        categorical_columns = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]
        
        for col in categorical_columns:
            df_cleaned[col] = df_cleaned[col].astype('category')
        
        self.logger.info(f"Data cleaned. New shape: {df_cleaned.shape}")
        return df_cleaned
    
    def data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        summary = {
            'total_records': len(df),
            'unique_customers': df['customerID'].nunique(),
            'churn_rate': df['Churn'].mean(),
            'demographic_summary': {
                'gender_distribution': self._safe_value_counts(df['gender']),
                'senior_citizen_rate': df['SeniorCitizen'].mean(),
                'partner_rate': df['Partner'].mean(),
                'dependents_rate': df['Dependents'].mean()
            },
            'service_summary': {
                'phone_service_rate': df['PhoneService'].mean(),
                'internet_service_distribution': self._safe_value_counts(df['InternetService']),
                'contract_distribution': self._safe_value_counts(df['Contract'])
            }
        }
        
        return summary
    
    def _safe_value_counts(self, series):
        
        
        if pd.api.types.is_categorical_dtype(series):
            series = series.astype(str)
        
        
        return series.value_counts(normalize=True).to_dict()
