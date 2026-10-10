import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class ChurnPredictorModel:
    def __init__(self, dataset_path="dataset/customer_churn.csv"):
        self.df = pd.read_csv(dataset_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._prepare_data()

    def _prepare_data(self):
        # Convert TotalCharges to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce').fillna(0)
        
        # Binary target encoding
        self.df['ChurnBinary'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                            'PaperlessBilling', 'PaymentMethod']
                            
        self.features_df = self.df.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.features_df[col] = le.fit_transform(self.features_df[col].astype(str))
            self.label_encoders[col] = le
            
        self.feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'] + categorical_cols
        
        X = self.features_df[self.feature_cols]
        y = self.features_df['ChurnBinary']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        preds = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, preds)

    def predict_single_customer(self, customer_dict):
        input_df = pd.DataFrame([customer_dict])
        
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0
                    
        X_input = input_df[self.feature_cols]
        X_scaled = self.scaler.transform(X_input)
        
        prob = float(self.model.predict_proba(X_scaled)[0][1])
        prediction = "CHURN RISK" if prob >= 0.45 else "RETAINED / LOW RISK"
        
        return {
            'churn_probability': round(prob * 100, 1),
            'prediction': prediction,
            'model_accuracy': round(self.accuracy * 100, 1)
        }
