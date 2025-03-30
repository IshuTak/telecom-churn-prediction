import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from typing import Tuple, Dict
import logging

class ChurnPredictor:
    def __init__(self, customer_features: pd.DataFrame):
        """
        Initialize Churn Predictor
        
        Args:
            customer_features (pd.DataFrame): Customer-level features
        """
        self.logger = logging.getLogger(__name__)
        self.features = customer_features.copy()
    
    def prepare_churn_features(self, preprocessor):
        """
        Prepare features for churn prediction
        
        Args:
            preprocessor: Feature preprocessing transformer
        
        Returns:
            Prepared features and target
        """
        # Select features
        features_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure', 'MonthlyCharges', 'TotalCharges'
        ]
        
        X = self.features[features_columns].copy()
        
        # Create tenure category
        X['tenure_category'] = pd.cut(
            X['tenure'], 
            bins=[0, 12, 24, 36, 48, 60, np.inf], 
            labels=['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
        )
        
        # Create monthly charges category
        X['monthly_charges_category'] = pd.qcut(
            X['MonthlyCharges'], 
            q=4, 
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # Create total services column
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        def count_services(row):
            return sum(1 for col in service_columns if row[col] in [1, 'Yes'])
        
        X['total_services'] = X.apply(count_services, axis=1)
        
        # Create customer value score
        X['customer_value_score'] = (
            X['tenure'] * 
            X['total_services'] * 
            (X['MonthlyCharges'] / 100)
        )
        
        # Create churn risk score
        def calculate_churn_risk(row):
            risk_score = 0
            if row['tenure'] < 12:
                risk_score += 1
            if row['MonthlyCharges'] > X['MonthlyCharges'].quantile(0.75):
                risk_score += 1
            if row['total_services'] < 2:
                risk_score += 1
            if row['Contract'] in ['Month-to-month', 1]:
                risk_score += 1
            return risk_score
        
        X['churn_risk_score'] = X.apply(calculate_churn_risk, axis=1)
        
        y = self.features['Churn']
        
        # Preprocess features
        X_processed = preprocessor.fit_transform(X)
        
        return X_processed, y
    
    def build_churn_model(self, X, y, preprocessor):
        """
        Build and evaluate churn prediction models
        
        Args:
            X: Processed features
            y: Target variable
            preprocessor: Feature preprocessor
        
        Returns:
            Tuple of models and performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest Model
        rf_model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # XGBoost Model
        xgb_model = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        # Performance metrics
        def get_model_metrics(model, X_test, y_test, preprocessor):
            y_pred = model.predict(X_test)
            
            # Robust feature names extraction
            try:
                # Try newer scikit-learn method
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                try:
                    # Try alternative method
                    feature_names = (
                        preprocessor.named_transformers_['cat'].get_feature_names_out().tolist() + 
                        preprocessor.named_transformers_['num'].get_feature_names_out().tolist()
                    )
                except Exception:
                    # Fallback to generic feature names
                    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
            
            return {
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'feature_importances': dict(zip(
                    feature_names, 
                    model.feature_importances_
                ))
            }
        
        rf_metrics = get_model_metrics(rf_model, X_test, y_test, preprocessor)
        xgb_metrics = get_model_metrics(xgb_model, X_test, y_test, preprocessor)
        
        return {
            'random_forest': rf_model,
            'xgboost': xgb_model,
            'rf_metrics': rf_metrics,
            'xgb_metrics': xgb_metrics
        }