import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict
import logging

class TelecomSegmentation:
    def __init__(self, customer_features: pd.DataFrame):
        
        self.logger = logging.getLogger(__name__)
        self.features = customer_features.copy()
    
    def advanced_segmentation(self) -> Tuple[pd.DataFrame, Dict]:
        
        clustering_features = [
            'tenure', 
            'MonthlyCharges', 
            'TotalCharges', 
            'total_services',
            'customer_value_score'
        ]
        
        
        X = self.features[clustering_features]
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
       
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.features['Customer_Segment'] = kmeans.fit_predict(X_scaled)
        
        
        segment_profile = self.features.groupby('Customer_Segment').agg({
            'tenure': ['mean', 'min', 'max'],
            'MonthlyCharges': ['mean', 'min', 'max'],
            'TotalCharges': ['mean', 'min', 'max'],
            'total_services': ['mean', 'count'],
            'Churn': 'mean'
        })
        
        return self.features, segment_profile
