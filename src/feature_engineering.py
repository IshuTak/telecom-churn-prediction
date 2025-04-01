import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TelecomFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        
        self.df = df
    
    def create_advanced_features(self) -> pd.DataFrame:
        
        
        df_features = self.df.copy()
        
        
        df_features['tenure_category'] = pd.cut(
            df_features['tenure'], 
            bins=[0, 12, 24, 36, 48, 60, np.inf], 
            labels=['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
        )
        
        
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        
        def count_services(row):
            return sum(1 for col in service_columns if row[col] in [1, 'Yes'])
        
        df_features['total_services'] = df_features.apply(count_services, axis=1)
        
        
        df_features['monthly_charges_category'] = pd.qcut(
            df_features['MonthlyCharges'], 
            q=4, 
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        
        df_features['customer_value_score'] = (
            df_features['tenure'] * 
            df_features['total_services'] * 
            (df_features['MonthlyCharges'] / 100)
        )
        
        
        def calculate_churn_risk(row):
            risk_score = 0
            if row['tenure'] < 12:
                risk_score += 1
            if row['MonthlyCharges'] > df_features['MonthlyCharges'].quantile(0.75):
                risk_score += 1
            if row['total_services'] < 2:
                risk_score += 1
            if row['Contract'] in ['Month-to-month', 1]:
                risk_score += 1
            return risk_score
        
        df_features['churn_risk_score'] = df_features.apply(calculate_churn_risk, axis=1)
        
        return df_features
    
    def prepare_model_features(self, df):
        
        categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure_category', 'monthly_charges_category'
        ]
        
        numerical_features = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 
            'total_services', 'customer_value_score', 
            'churn_risk_score'
        ]
        
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
            ])
        
        return preprocessor
