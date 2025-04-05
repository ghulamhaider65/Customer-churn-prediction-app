# Customer-churn-prediction-app

# Customers-Churm-prediction-System
A machine learning-based solution to predict customer churn for telecom companies. This system helps businesses identify at-risk customers and take proactive retention measures.

```
TELCO CHURN PREDICTION WORKFLOW
├── Data Pipeline
│   ├── Input: WA_Fn-UseC_-Telco-Customer-Churn2.csv
│   ├── Data Cleaning (handle missing values, type conversion)
│   └── Feature Engineering (encoding, transformations)
│
├── ML Pipeline
│   ├── Feature Selection (SelectKBest)
│   ├── Model Training (XGBoost with GridSearchCV)
│   ├── Evaluation (Accuracy, Precision, Recall, ROC-AUC)
│   └── Model Persistence (joblib)
│
├── Prediction Services
│   ├── Single Prediction (form-based input)
│   └── Bulk Prediction (CSV processing)
│
└── Visualization Layer
    ├── Interactive Dashboards
    │   ├── Churn Distribution
    │   ├── Feature Importance
    │   └── Performance Metrics
    │
    └── Real-time Outputs
        ├── Probability Gauges
        └── Prediction Tables
```

KEY COMPONENTS:
1. app.py - Streamlit application controller
2. ml_pipeline.py - Core machine learning operations
3. style.css - UI styling
4. churn_model.pkl - Serialized trained model
5. requirements.txt - Dependency specification

DATA FLOW:
Raw Data → Preprocessing → Feature Selection → Model Training → 
Evaluation → Prediction Interface → Visualizations 
