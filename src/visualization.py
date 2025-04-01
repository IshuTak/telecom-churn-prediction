import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import logging
import os
from typing import Optional

class CustomerInsightVisualizer:
    def __init__(self, segmented_customers: pd.DataFrame):
        
        self.logger = logging.getLogger(__name__)
        
        
        if 'Customer_Segment' not in segmented_customers.columns:
            segmented_customers['Customer_Segment'] = 0
        
        self.data = segmented_customers
    
    def segment_distribution_plot(self, save_path: str = 'visualizations/segment_distribution.png'):
       
        try:
            plt.figure(figsize=(12, 6))
            segment_counts = self.data['Customer_Segment'].value_counts()
            
            
            ax = segment_counts.plot(kind='bar')
            plt.title('Customer Segment Distribution', fontsize=15)
            plt.xlabel('Customer Segment', fontsize=12)
            plt.ylabel('Number of Customers', fontsize=12)
            plt.tight_layout()
            
            
            total = len(self.data)
            for i, v in enumerate(segment_counts):
                percentage = v / total * 100
                ax.text(i, v, f'{percentage:.1f}%', 
                        ha='center', va='bottom')
            
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Segment distribution plot saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating segment distribution plot: {e}")
    
    def segment_churn_analysis(self, save_path: str = 'visualizations/segment_churn_analysis.png'):
        
        try:
            plt.figure(figsize=(12, 6))
            churn_by_segment = self.data.groupby('Customer_Segment')['Churn'].mean()
            
            ax = churn_by_segment.plot(kind='bar')
            plt.title('Churn Rate by Customer Segment', fontsize=15)
            plt.xlabel('Customer Segment', fontsize=12)
            plt.ylabel('Churn Rate', fontsize=12)
            plt.tight_layout()
            
            # Add percentage labels
            for i, v in enumerate(churn_by_segment):
                ax.text(i, v, f'{v*100:.1f}%', 
                        ha='center', va='bottom')
            
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Segment churn analysis plot saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating segment churn analysis plot: {e}")
    
    def service_usage_heatmap(self, save_path: str = 'visualizations/service_usage_heatmap.png'):
        
        try:
            # Select service columns
            service_columns = [
                'PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            
            
            service_usage = self.data.groupby('Customer_Segment')[service_columns].mean()
            
            plt.figure(figsize=(15, 8))
            sns.heatmap(service_usage, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Service Usage Across Customer Segments', fontsize=15)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Service usage heatmap saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating service usage heatmap: {e}")
    
    def interactive_segment_analysis(self, save_path: str = 'visualizations/interactive_segments.html'):
        
        try:
           
            fig = px.scatter(
                self.data, 
                x='tenure', 
                y='MonthlyCharges',
                color='Customer_Segment',
                size='TotalCharges',
                hover_data=['Churn', 'Contract', 'InternetService'],
                title='Customer Segments: Tenure vs Monthly Charges',
                labels={
                    'tenure': 'Customer Tenure (Months)',
                    'MonthlyCharges': 'Monthly Charges',
                    'Customer_Segment': 'Customer Segment'
                }
            )
            
            
            fig.update_layout(
                xaxis_title='Customer Tenure (Months)',
                yaxis_title='Monthly Charges',
                legend_title='Customer Segment'
            )
            
            
            fig.write_html(save_path)
            self.logger.info(f"Interactive segment analysis saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating interactive segment analysis: {e}")
    
    def churn_factors_correlation(self, save_path: str = 'visualizations/churn_correlation.png'):
        
        try:
            # Select relevant features
            churn_features = [
                'tenure', 
                'MonthlyCharges', 
                'TotalCharges', 
                'Churn',
                'total_services',
                'customer_value_score'
            ]
            
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.data[churn_features].corr()
            
            
            sns.heatmap(
                correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5
            )
            
            plt.title('Correlation of Churn-Related Features', fontsize=15)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Churn factors correlation plot saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating churn factors correlation plot: {e}")
    
    def generate_comprehensive_visualizations(self):
        
        try:
            # Ensure visualizations directory exists
            os.makedirs('visualizations', exist_ok=True)
            
            # Create various visualizations
            self.segment_distribution_plot('visualizations/segment_distribution.png')
            self.segment_churn_analysis('visualizations/segment_churn_analysis.png')
            self.service_usage_heatmap('visualizations/service_usage_heatmap.png')
            self.interactive_segment_analysis('visualizations/interactive_segments.html')
            self.churn_factors_correlation('visualizations/churn_correlation.png')
            
            self.logger.info("All comprehensive visualizations generated successfully")
        except Exception as e:
            self.logger.error(f"Error generating comprehensive visualizations: {e}")

class ChurnVisualization:
    def __init__(self, churn_data: pd.DataFrame):
        
        self.logger = logging.getLogger(__name__)
        self.data = churn_data
    
    def churn_risk_distribution(self, save_path: str = 'visualizations/churn_risk_distribution.png'):
        
        try:
            plt.figure(figsize=(10, 6))
            churn_counts = self.data['Churn'].value_counts()
            
            
            plt.pie(
                churn_counts, 
                labels=['Retained', 'Churned'], 
                autopct='%1.1f%%',
                colors=['#2196F3', '#F44336']
            )
            plt.title('Customer Churn Distribution', fontsize=15)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Churn distribution plot saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating churn distribution plot: {e}")
    
    def contract_churn_analysis(self, save_path: str = 'visualizations/contract_churn_analysis.png'):
        
        try:
            plt.figure(figsize=(12, 6))
            churn_by_contract = self.data.groupby('Contract')['Churn'].mean()
            
            ax = churn_by_contract.plot(kind='bar')
            plt.title('Churn Rate by Contract Type', fontsize=15)
            plt.xlabel('Contract Type', fontsize=12)
            plt.ylabel('Churn Rate', fontsize=12)
            plt.tight_layout()
            
            
            for i, v in enumerate(churn_by_contract):
                ax.text(i, v, f'{v*100:.1f}%', 
                        ha='center', va='bottom')
            
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Contract churn analysis plot saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error creating contract churn analysis plot: {e}")
