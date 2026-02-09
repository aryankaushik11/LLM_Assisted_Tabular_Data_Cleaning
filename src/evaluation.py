import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    def evaluate_quality(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
        """
        Compares static data quality metrics.
        """
        raw_missing = df_raw.isnull().sum().sum()
        clean_missing = df_clean.isnull().sum().sum()
        
        raw_rows = len(df_raw)
        clean_rows = len(df_clean)
        
        return {
            "missing_values_reduction": {
                "raw": int(raw_missing),
                "clean": int(clean_missing),
                "reduction_pct": round((raw_missing - clean_missing) / raw_missing * 100, 2) if raw_missing > 0 else 0
            },
            "rows_retained": {
                "raw": raw_rows,
                "clean": clean_rows,
                "retention_pct": round(clean_rows / raw_rows * 100, 2) if raw_rows > 0 else 0
            }
        }

    def evaluate_ml_performance(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame, target_col: str) -> dict:
        """
        Trains a simple model on Raw (imputed simply) vs Cleaned data to measure lift.
        """
        if target_col not in df_raw.columns or target_col not in df_clean.columns:
            logger.warning(f"Target column {target_col} not found. Skipping ML evaluation.")
            return {}
            
        def prepare_data(df, target):
            # Drop rows where target is missing
            df = df.dropna(subset=[target])
            X = df.drop(columns=[target])
            y = df[target]
            
            # Simple preprocessing for the "Raw" and "Clean" comparison
            # We encode strings to numbers so the model runs
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                # Handle unknown values/NaNs by converting to string
                X[col] = X[col].astype(str)
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
                
            # Impute remaining NaNs with mean (for raw data mainly)
            imputer = SimpleImputer(strategy='most_frequent')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            # Encode target
            y = LabelEncoder().fit_transform(y.astype(str))
            
            return X, y

        try:
            logger.info("Training ML model on RAW data...")
            X_raw, y_raw = prepare_data(df_raw, target_col)
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
            clf_raw = RandomForestClassifier(n_estimators=50, random_state=42)
            clf_raw.fit(X_train_r, y_train_r)
            acc_raw = accuracy_score(y_test_r, clf_raw.predict(X_test_r))
            
            logger.info("Training ML model on CLEAN data...")
            X_clean, y_clean = prepare_data(df_clean, target_col)
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
            clf_clean = RandomForestClassifier(n_estimators=50, random_state=42)
            clf_clean.fit(X_train_c, y_train_c)
            acc_clean = accuracy_score(y_test_c, clf_clean.predict(X_test_c))
            
            return {
                "accuracy_raw": round(acc_raw, 4),
                "accuracy_clean": round(acc_clean, 4),
                "improvement": round((acc_clean - acc_raw) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"ML Evaluation failed: {e}")
            return {"error": str(e)}
