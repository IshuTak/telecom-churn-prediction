# 🚀 Telecom Customer Churn Prediction Project

## Project Overview

This advanced data science project focuses on predicting customer churn in the telecom industry using sophisticated machine learning techniques. By leveraging comprehensive data analysis and predictive modeling, we provide actionable insights to help telecom companies reduce customer attrition.


## 📊 Key Metrics and Achievements

### Model Performance
- **XGBoost Model Accuracy**: 78.68%
- **Random Forest Model Accuracy**: 77.61%
- **Churn Prediction Precision**: 0.614
- **Churn Prediction Recall**: 0.535

## 🔍 Project Highlights

### 1. Customer Segmentation
We developed a sophisticated customer segmentation strategy revealing distinct customer groups:

![segment_distribution](https://github.com/user-attachments/assets/661816be-94a3-44e8-a5f1-c0f663a316e7)


**Segment Breakdown**:
- Segment 0: 31.9% of customers (13.5% Churn Rate)
- Segment 1: 30.8% of customers (14.6% Churn Rate)
- Segment 2: 22.9% of customers (19.5% Churn Rate)
- Segment 3: 14.4% of customers (49.1% Churn Rate) 🚨 Highest Risk Segment

### 2. Churn Risk Analysis
Comprehensive analysis of churn risk across different customer segments:

![segment_churn_analysis](https://github.com/user-attachments/assets/36ec80aa-d14a-45e5-ac64-aa5e92a3b3cc)

### 3. Correlation of Churn-Related Features
Deep dive into feature relationships:

![churn_correlation](https://github.com/user-attachments/assets/70c3b7c3-1cdf-4add-9f0b-3e20293282ec)

### 4. Service Usage Insights
Understanding service penetration across segments:

![service_usage_heatmap](https://github.com/user-attachments/assets/532ff835-5fdc-45bf-87a7-551e340c54cf)

## 🛠 Technical Approach

### Methodology
1. **Data Preprocessing**
   - Comprehensive data cleaning
   - Handling missing values
   - Feature engineering

2. **Feature Engineering**
   - Created advanced features
   - Calculated customer value scores
   - Developed churn risk indicators

3. **Machine Learning Models**
   - Random Forest Classifier
   - XGBoost Classifier
   - Advanced hyperparameter tuning
   - Cross-validation techniques

## 💡 Key Insights

### Churn Risk Factors
1. Short customer tenure
2. High monthly charges
3. Month-to-month contracts
4. Lack of additional services
5. No technical support

### Recommended Actions
- Develop targeted retention strategies
- Create personalized service packages
- Implement proactive customer engagement
- Offer competitive long-term contracts
- Enhance technical support services

## 🧰 Technologies and Tools

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3.3-yellow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5.0-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.3-lightgrey)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/telecom-churn-prediction.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
