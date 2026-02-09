import pandas as pd
import numpy as np
import json

class DataProfiler:
    """
    Analyzes a DataFrame and outputs a JSON summary of its statistical properties,
    data types, and potential quality issues.
    """
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Generates a comprehensive profile of the dataset.
        """
        profile = {
            "rows": len(df),
            "columns": list(df.columns),
            "column_analysis": {},
            "overall_health": {}
        }
        
        for col in df.columns:
            col_data = df[col]
            dtype = str(col_data.dtype)
            
            # Basic calculation
            missing_count = int(col_data.isnull().sum())
            missing_ratio = float(missing_count / len(df)) if len(df) > 0 else 0
            unique_count = int(col_data.nunique())
            
            col_stats = {
                "dtype": dtype,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 4),
                "unique_count": unique_count,
                "sample_values": col_data.dropna().sample(min(5, len(col_data))).tolist() if len(col_data) > 0 else []
            }

            # Type specific analysis
            if np.issubdtype(col_data.dtype, np.number):
                col_stats.update({
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "std": float(col_data.std()) if not col_data.empty else None,
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "zeros_count": int((col_data == 0).sum())
                })
                # Outlier detection (Z-score > 3)
                if col_stats["std"] and col_stats["std"] > 0:
                    z_scores = np.abs((col_data - col_stats["mean"]) / col_stats["std"])
                    outliers = int((z_scores > 3).sum())
                    col_stats["potential_outliers"] = outliers
                else:
                    col_stats["potential_outliers"] = 0

            else:
                # Categorical analysis
                try:
                    top_values = col_data.value_counts().head(5).to_dict()
                    col_stats["top_values"] = {k: int(v) for k, v in top_values.items()}
                except Exception:
                    col_stats["top_values"] = {}
                
                # Semantic check candidates (if low quantity of unique values but object type)
                if unique_count < 50 and unique_count > 0:
                    col_stats["is_categorical"] = True
            
            profile["column_analysis"][col] = col_stats

        return profile

    def to_json(self, profile: dict, indent: int = 2) -> str:
        """
        Converts the profile dictionary to a JSON string.
        """
        # handling non-serializable types if any remain
        def default_serializer(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return str(obj)
            
        return json.dumps(profile, indent=indent, default=default_serializer)
