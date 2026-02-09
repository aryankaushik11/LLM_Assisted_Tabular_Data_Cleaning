import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Executor:
    """
    Executes the JSON workflow on the DataFrame.
    """
    
    def execute(self, df: pd.DataFrame, workflow: dict) -> pd.DataFrame:
        """
        Applies transformations step-by-step.
        """
        df_clean = df.copy()
        
        steps = workflow.get("steps", [])
        logger.info(f"Executing {len(steps)} cleaning steps.")
        
        for step in steps:
            try:
                op = step.get("operation")
                col = step.get("column")
                params = step.get("params", {})
                
                logger.info(f"Step {step.get('step_id')}: {op} on {col}")
                
                if op == "drop_na":
                    if col:
                        df_clean.dropna(subset=[col], inplace=True)
                    else:
                        df_clean.dropna(inplace=True)
                        
                elif op == "fill_na":
                    strategy = params.get("strategy")
                    value = params.get("value")
                    if col in df_clean.columns:
                        if strategy == "mean":
                            fill_val = df_clean[col].mean()
                        elif strategy == "median":
                            fill_val = df_clean[col].median()
                        elif strategy == "mode":
                            fill_val = df_clean[col].mode()[0]
                        else:
                            fill_val = value
                        df_clean[col].fillna(fill_val, inplace=True)
                        
                elif op == "drop_duplicates":
                    df_clean.drop_duplicates(inplace=True)
                    
                elif op == "cast_type":
                    dtype = params.get("dtype")
                    if col in df_clean.columns:
                        if dtype == "int":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
                        elif dtype == "float":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        elif dtype == "datetime":
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        else:
                            df_clean[col] = df_clean[col].astype(dtype)

                elif op == "replace":
                    old_val = params.get("old")
                    new_val = params.get("new")
                    if col in df_clean.columns:
                        # Handle specific "NaN" string requests
                        if new_val == "NaN":
                            new_val = np.nan
                        df_clean[col].replace(old_val, new_val, inplace=True)
                        
                elif op == "rename":
                    new_name = params.get("new_name")
                    if col in df_clean.columns:
                        df_clean.rename(columns={col: new_name}, inplace=True)

                elif op == "remove_outliers":
                    method = params.get("method", "z-score")
                    threshold = params.get("threshold", 3)
                    if col in df_clean.columns and method == "z-score":
                        mean = df_clean[col].mean()
                        std = df_clean[col].std()
                        z_scores = np.abs((df_clean[col] - mean) / std)
                        df_clean = df_clean[z_scores < threshold]
                        
            except Exception as e:
                logger.error(f"Failed to execute step {step}: {e}")
                
        return df_clean
