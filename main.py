import logging
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_insights.log'),
        logging.StreamHandler()
    ]
)

# Import custom modules
from src.data_preprocessing import TelecomDataPreprocessor
from src.feature_engineering import TelecomFeatureEngineer
from src.customer_segmentation import TelecomSegmentation
from src.churn_prediction import ChurnPredictor
from src.visualization import CustomerInsightVisualizer

def create_output_directories():
    """Create necessary output directories"""
    directories = [
        'outputs', 
        'reports', 
        'visualizations', 
        'models/saved_models',
          
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)



def main():
    try:
        # Create output directories
        create_output_directories()
        
        # Data Preprocessing
        preprocessor = TelecomDataPreprocessor('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        cleaned_data = preprocessor.clean_data()
        data_summary = preprocessor.data_summary(cleaned_data)
        logging.info(f"Data Summary: {data_summary}")
        
        # Feature Engineering
        feature_engineer = TelecomFeatureEngineer(cleaned_data)
        advanced_features = feature_engineer.create_advanced_features()
        
        # Prepare model features
        model_preprocessor = feature_engineer.prepare_model_features(advanced_features)
        
        # Customer Segmentation
        segmentation = TelecomSegmentation(advanced_features)
        segmented_customers, segment_profile = segmentation.advanced_segmentation()
        
        # Churn Prediction
        churn_predictor = ChurnPredictor(advanced_features)
        X_processed, y = churn_predictor.prepare_churn_features(model_preprocessor)
        
        # Build churn models (pass preprocessor)
        churn_models = churn_predictor.build_churn_model(X_processed, y, model_preprocessor)
        
        # Visualization
        visualizer = CustomerInsightVisualizer(segmented_customers)
        visualizer.generate_comprehensive_visualizations()
        
        # Save models and results
        joblib.dump(churn_models['random_forest'], 'models/saved_models/rf_churn_model.pkl')
        joblib.dump(churn_models['xgboost'], 'models/saved_models/xgb_churn_model.pkl')
        
        # Log model performance
        logging.info("Random Forest Metrics:")
        logging.info(churn_models['rf_metrics']['classification_report'])
        
        logging.info("XGBoost Metrics:")
        logging.info(churn_models['xgb_metrics']['classification_report'])
        
        print("Telecom Customer Insights Analysis Complete!")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()