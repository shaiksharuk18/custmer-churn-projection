# 📊 Customer Churn Prediction & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-222222.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Customer Churn Prediction & Retention Analytics** web application built with **Python**, **Scikit-Learn**, **XGBoost**, and **Streamlit**, inspired by **CampusX Machine Learning Projects**.

This system analyzes customer demographics, contract types, tenure, and service usage to predict churn probability in real time, helping businesses prevent customer attrition.

---

## 🌟 Key Capabilities

- **📊 Exploratory Churn Analytics**: Analyzes contract type impact, tenure decay, and monthly billing metrics.
- **🤖 Machine Learning Pipeline**: Feature encoding (`LabelEncoder`), StandardScaler normalization, SMOTE class balancing, and **Random Forest / XGBoost** classification.
- **🎛️ Interactive Streamlit Dashboard**: Real-time customer churn probability gauge, risk classification, and retention strategy suggestions (`app.py`).
- **🎯 Real-Time Risk Scoring**: Sub-second prediction latency for live customer service calls.

---

## 📁 Repository Structure

```
Customer-Churn-Prediction/
├── app.py                     # Streamlit web application & UI
├── src/
│   └── model_trainer.py       # ML preprocessing & Random Forest / XGBoost engine
├── dataset/
│   └── customer_churn.csv     # Telco customer churn dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Slack-Hacker/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

Navigate to **`http://localhost:8501`** in your browser.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
