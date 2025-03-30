import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict
import logging

class TelecomSegmentation:
    def __init__(self, customer_features: pd.DataFrame):
        """
        Initialize Telecom Customer Segmentation
        
        Args:
            customer_features (pd.DataFrame): Customer-level features
        """
        self.logger = logging.getLogger(__name__)
        self.features = customer_features.copy()
    
    def advanced_segmentation(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Advanced customer segmentation
        
        Returns:
            Tuple of segmented customers and segment profile
        """
        # Select features for clustering
        clustering_features = [
            'tenure', 
            'MonthlyCharges', 
            'TotalCharges', 
            'total_services',
            'customer_value_score'
        ]
        
        # Prepare features
        X = self.features[clustering_features]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-Means Clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.features['Customer_Segment'] = kmeans.fit_predict(X_scaled)
        
        # Create segment profile
        segment_profile = self.features.groupby('Customer_Segment').agg({
            'tenure': ['mean', 'min', 'max'],
            'MonthlyCharges': ['mean', 'min', 'max'],
            'TotalCharges': ['mean', 'min', 'max'],
            'total_services': ['mean', 'count'],
            'Churn': 'mean'
        })
        
        return self.features, segment_profile