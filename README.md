# 💳 Credit Card Fraud Detection System  
### End-to-End Machine Learning Pipeline for Fraud Detection

A production-ready machine learning system for detecting fraudulent credit card transactions using probability calibration, threshold optimization, and explainable AI techniques.

---

## 📌 Executive Summary

This project builds a comprehensive fraud detection system that helps financial teams:

- Detect fraudulent transactions early  
- Minimize false negatives (missed fraud)  
- Minimize false positives (customer inconvenience)  
- Prioritize recall and ROC-AUC due to extreme class imbalance  

Instead of optimizing only for accuracy, the system aligns model decisions with **financial risk mitigation** objectives.

---

## 🧠 Problem Statement

Credit card fraud leads to:

- Financial losses for banks and merchants  
- Customer distrust  
- Operational inefficiencies  

Challenges addressed:

- Extremely imbalanced dataset (~0.17% fraud cases)  
- Need for high recall and robust ROC-AUC  
- Feature-level interpretability  

Techniques used:

- Stratified cross-validation  
- SMOTE oversampling  
- Threshold optimization  
- Feature importance analysis  
- Ensemble modeling (Random Forest, XGBoost)

---

## 🏗️ Project Architecture
---
Raw Transaction Data
↓
EDA & Feature Engineering
↓
Preprocessing (Scaling)
↓
Train/Test Split (Stratified)
↓
SMOTE Resampling (on training set)
↓
Model Training (Logistic Regression / Random Forest / XGBoost)
↓
Threshold Optimization
↓
Evaluation (F1, Recall, ROC-AUC)
↓
Feature Importance & Deployment
---

## 🤖 Models Evaluated

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline, interpretable |
| Random Forest | Ensemble, handles non-linearities |
| XGBoost | Boosted tree model, final selected model |

---

## 📊 Evaluation Metrics

Because fraud detection is highly imbalanced, we prioritized:

- **Recall** (capture as many frauds as possible)  
- **F1 Score** (balance precision and recall)  
- **ROC-AUC** (ranking fraud probability)  
- Precision–Recall curves  
- Confusion matrices  

Accuracy alone is **misleading** due to the class imbalance.

---

## 🔹 Key EDA Insights

- Fraud cases are extremely rare (~0.17%)  
- Fraud transaction amounts differ from normal transactions  
- Fraud occurs more frequently at specific times (hour-of-day patterns)  
- Feature distributions indicate some PCA-transformed features (V10, V12, V14) are informative  

---

## 💻 Feature Engineering & Preprocessing

- **Scaling:** `Time` and `Amount` standardized using `StandardScaler`  
- **Train/Test Split:** 80/20 stratified split  
- **SMOTE:** Oversampling applied to training set only  
- **Feature Selection:** All original PCA features retained  

---

## 📈 Model Performance

### Before SMOTE

| Model | Recall | F1 Score | ROC-AUC |
|-------|--------|----------|---------|
| Logistic Regression | 0.64 | 0.55 | 0.94 |
| Random Forest | 0.72 | 0.61 | 0.97 |
| XGBoost | 0.70 | 0.62 | 0.97 |

### After SMOTE

| Model | Recall | F1 Score | ROC-AUC |
|-------|--------|----------|---------|
| Logistic Regression | 0.84 | 0.70 | 0.96 |
| Random Forest | 0.89 | 0.75 | 0.98 |
| XGBoost | 0.87 | 0.73 | 0.97 |

✔ Random Forest selected as **final model** for deployment.

---

## 💰 Business Optimization Layer

### 🎯 Threshold Optimization

- Default threshold = 0.5 often misses frauds  
- Optimized thresholds (e.g., 0.3) improve **recall** while maintaining precision  
- Enables actionable fraud alerts without overwhelming false positives  

---

## 🔍 Feature Importance

Top 10 features for Random Forest:

1. V14  
2. V12  
3. V10  
4. Amount  
5. V17  
6. V16  
7. V11  
8. V3  
9. V7  
10. V2  

Visualization used **barplots** for clear interpretability.

---

## 📁 Project Structure
---
creditcard-fraud-detection/
│
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── outputs/
├── requirements.txt
└── README.md
---

## 📌 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
---
