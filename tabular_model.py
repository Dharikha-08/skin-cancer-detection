import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import os

class TabularModel:
    def __init__(self, model_path="best_tabular_model.pkl", imbalance_ratio=20):
        self.model_path = model_path
        # Optimized XGBoost parameters for high clinical precision
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            scale_pos_weight=imbalance_ratio,
            eval_metric='logloss',
            random_state=42
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        
        # Streamlined feature selection (ONLY what the user requested)
        self.categorical_cols = [
            'pigment_network', 'streaks', 'pigmentation', 
            'elevation', 'location', 'sex'
        ]
        self.numerical_cols = [] # Removed 7-point score
        self.feature_cols = self.categorical_cols + self.numerical_cols
        self.is_fitted = False

    def preprocess(self, df, fit=False):
        # Filter only available columns
        available_cols = [c for c in self.feature_cols if c in df.columns]
        df_copy = df[available_cols].copy()
        
        # Ensure all columns exist
        for col in self.feature_cols:
            if col not in df_copy.columns:
                df_copy[col] = "absent" if col in self.categorical_cols else 0

        # Handle missing values
        if fit:
            if self.numerical_cols:
                df_copy[self.numerical_cols] = self.imputer_num.fit_transform(df_copy[self.numerical_cols])
            df_copy[self.categorical_cols] = self.imputer_cat.fit_transform(df_copy[self.categorical_cols])
        else:
            if self.numerical_cols:
                df_copy[self.numerical_cols] = self.imputer_num.transform(df_copy[self.numerical_cols])
            df_copy[self.categorical_cols] = self.imputer_cat.transform(df_copy[self.categorical_cols])

        # Encode categorical variables
        for col in self.categorical_cols:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
            if fit:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                le_classes = [str(c).lower().strip() for c in le.classes_]
                def robust_encode(val):
                    v = str(val).lower().strip()
                    if v in le_classes:
                        return le.transform([le.classes_[le_classes.index(v)]])[0]
                    return 0
                df_copy[col] = df_copy[col].apply(robust_encode)

        # Normalize
        if fit and self.numerical_cols:
            df_copy[self.numerical_cols] = self.scaler.fit_transform(df_copy[self.numerical_cols])
        elif not fit and self.numerical_cols:
            df_copy[self.numerical_cols] = self.scaler.transform(df_copy[self.numerical_cols])

        return df_copy[self.feature_cols]

    def train(self, df_train, labels_train):
        print(f"Training Streamlined Tabular Model (XGBoost)...")
        X_train = self.preprocess(df_train, fit=True)
        self.model.fit(X_train, labels_train)
        self.is_fitted = True
        self.save_model()
        print("Model trained using only specified clinical fields.")

    def predict_proba(self, df):
        if not self.is_fitted:
            self.load_model()
        X = self.preprocess(df)
        return self.model.predict_proba(X)[:, 1]

    def save_model(self):
        data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputer_num': self.imputer_num,
            'imputer_cat': self.imputer_cat,
            'is_fitted': self.is_fitted
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.label_encoders = data['label_encoders']
                self.scaler = data['scaler']
                self.imputer_num = data['imputer_num']
                self.imputer_cat = data['imputer_cat']
                self.is_fitted = data['is_fitted']
            print("XGBoost model loaded successfully.")
        else:
            print("Warning: No model found.")
