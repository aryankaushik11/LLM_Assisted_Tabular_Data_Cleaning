import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    def evaluate_quality(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
        """
        Compares static data quality metrics across all 5 cleaning categories.
        """
        raw_missing = df_raw.isnull().sum().sum()
        clean_missing = df_clean.isnull().sum().sum()
        
        raw_rows = len(df_raw)
        clean_rows = len(df_clean)
        
        raw_cols = len(df_raw.columns)
        clean_cols = len(df_clean.columns)
        
        raw_duplicates = int(df_raw.duplicated().sum())
        clean_duplicates = int(df_clean.duplicated().sum())
        
        # Per-column missing value comparison
        column_missing_comparison = {}
        for col in df_raw.columns:
            raw_col_missing = int(df_raw[col].isnull().sum())
            clean_col_missing = int(df_clean[col].isnull().sum()) if col in df_clean.columns else "column_dropped"
            column_missing_comparison[col] = {
                "raw_missing": raw_col_missing,
                "clean_missing": clean_col_missing
            }
        
        # Check for new columns added (from split operations)
        new_columns = [col for col in df_clean.columns if col not in df_raw.columns]
        dropped_columns = [col for col in df_raw.columns if col not in df_clean.columns]
        
        return {
            "category_1_null_treatment": {
                "total_missing_raw": int(raw_missing),
                "total_missing_clean": int(clean_missing),
                "reduction_pct": round((raw_missing - clean_missing) / raw_missing * 100, 2) if raw_missing > 0 else 0,
                "column_details": column_missing_comparison
            },
            "category_2_duplicate_treatment": {
                "duplicates_raw": raw_duplicates,
                "duplicates_clean": clean_duplicates,
                "duplicates_removed": raw_duplicates - clean_duplicates
            },
            "category_3_missing_population": {
                "completeness_raw": round((1 - raw_missing / (raw_rows * raw_cols)) * 100, 2) if raw_rows * raw_cols > 0 else 0,
                "completeness_clean": round((1 - clean_missing / (clean_rows * clean_cols)) * 100, 2) if clean_rows * clean_cols > 0 else 0
            },
            "category_4_column_dropping": {
                "columns_raw": raw_cols,
                "columns_clean": clean_cols,
                "columns_dropped": dropped_columns,
                "columns_dropped_count": len(dropped_columns)
            },
            "category_5_column_splitting": {
                "new_columns_created": new_columns,
                "new_columns_count": len(new_columns)
            },
            "rows_retained": {
                "raw": raw_rows,
                "clean": clean_rows,
                "retention_pct": round(clean_rows / raw_rows * 100, 2) if raw_rows > 0 else 0
            }
        }

    def evaluate_ml_performance(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame, target_col: str) -> dict:
        """
        Trains a simple model on Raw vs Cleaned data to measure improvement.
        Auto-detects whether to use classification or regression based on target column.
        """
        if target_col not in df_raw.columns or target_col not in df_clean.columns:
            logger.warning(f"Target column '{target_col}' not found in one or both datasets. Skipping ML evaluation.")
            return {"skipped": True, "reason": f"Target column '{target_col}' not found"}
        
        # Determine task type based on target column characteristics
        target_dtype = df_raw[target_col].dtype
        n_unique = df_raw[target_col].nunique()
        n_rows = len(df_raw)
        
        # Skip if target is datetime - not suitable for direct ML
        if pd.api.types.is_datetime64_any_dtype(target_dtype):
            logger.info(f"Target '{target_col}' is datetime. Converting to numeric (hour) for regression.")
            # We'll convert to hour-of-day for a simple regression task
            task_type = "regression"
        elif n_unique <= 20 or (n_unique / n_rows < 0.05 and n_unique <= 50):
            task_type = "classification"
        elif np.issubdtype(target_dtype, np.number):
            task_type = "regression"
        else:
            # High-cardinality string column - try classification if reasonable
            if n_unique > 100:
                logger.warning(f"Target '{target_col}' has {n_unique} unique values. Too many for classification, skipping.")
                return {"skipped": True, "reason": f"Target has {n_unique} unique values, unsuitable for ML"}
            task_type = "classification"
        
        logger.info(f"ML task type: {task_type} (target='{target_col}', unique={n_unique}, dtype={target_dtype})")
            
        def prepare_data(df, target, is_datetime_target=False):
            df = df.dropna(subset=[target]).copy()
            X = df.drop(columns=[target])
            y = df[target].copy()
            
            # Handle datetime target: convert to numeric feature (e.g., minutes since midnight)
            if is_datetime_target or pd.api.types.is_datetime64_any_dtype(y):
                y = pd.to_datetime(y, errors='coerce')
                y = y.dt.hour * 60 + y.dt.minute  # minutes since midnight
                y = y.fillna(y.median())
            
            # Drop datetime columns from features (can't feed to sklearn)
            datetime_cols = X.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
            # Also detect string columns that look like times/dates
            for col in X.select_dtypes(include=['object']).columns:
                sample = X[col].dropna().head(20)
                try:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
                except (ValueError, TypeError):
                    pass
            if datetime_cols:
                X = X.drop(columns=datetime_cols, errors='ignore')
            
            # Encode string columns
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = X[col].astype(str)
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
                
            # Impute remaining NaNs
            imputer = SimpleImputer(strategy='most_frequent')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            if task_type == "classification":
                y = LabelEncoder().fit_transform(y.astype(str))
            else:
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.median())
            
            return X, y

        try:
            is_dt = pd.api.types.is_datetime64_any_dtype(target_dtype)
            
            if task_type == "classification":
                from sklearn.ensemble import RandomForestClassifier
                
                logger.info("Training classification model on RAW data...")
                X_raw, y_raw = prepare_data(df_raw, target_col, is_dt)
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
                clf_raw = RandomForestClassifier(n_estimators=50, random_state=42)
                clf_raw.fit(X_train_r, y_train_r)
                preds_raw = clf_raw.predict(X_test_r)
                acc_raw = accuracy_score(y_test_r, preds_raw)
                f1_raw = f1_score(y_test_r, preds_raw, average='weighted')
                
                logger.info("Training classification model on CLEAN data...")
                X_clean, y_clean = prepare_data(df_clean, target_col, is_dt)
                X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                clf_clean = RandomForestClassifier(n_estimators=50, random_state=42)
                clf_clean.fit(X_train_c, y_train_c)
                preds_clean = clf_clean.predict(X_test_c)
                acc_clean = accuracy_score(y_test_c, preds_clean)
                f1_clean = f1_score(y_test_c, preds_clean, average='weighted')
                
                return {
                    "task_type": "classification",
                    "accuracy_raw": round(acc_raw, 4),
                    "accuracy_clean": round(acc_clean, 4),
                    "accuracy_improvement": round((acc_clean - acc_raw) * 100, 2),
                    "f1_raw": round(f1_raw, 4),
                    "f1_clean": round(f1_clean, 4),
                    "f1_improvement": round((f1_clean - f1_raw) * 100, 2)
                }
            else:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import r2_score, mean_squared_error
                
                logger.info("Training regression model on RAW data...")
                X_raw, y_raw = prepare_data(df_raw, target_col, is_dt)
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
                reg_raw = RandomForestRegressor(n_estimators=50, random_state=42)
                reg_raw.fit(X_train_r, y_train_r)
                preds_raw = reg_raw.predict(X_test_r)
                r2_raw = r2_score(y_test_r, preds_raw)
                rmse_raw = mean_squared_error(y_test_r, preds_raw, squared=False)
                
                logger.info("Training regression model on CLEAN data...")
                X_clean, y_clean = prepare_data(df_clean, target_col, is_dt)
                X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                reg_clean = RandomForestRegressor(n_estimators=50, random_state=42)
                reg_clean.fit(X_train_c, y_train_c)
                preds_clean = reg_clean.predict(X_test_c)
                r2_clean = r2_score(y_test_c, preds_clean)
                rmse_clean = mean_squared_error(y_test_c, preds_clean, squared=False)
                
                return {
                    "task_type": "regression",
                    "r2_raw": round(r2_raw, 4),
                    "r2_clean": round(r2_clean, 4),
                    "r2_improvement": round((r2_clean - r2_raw) * 100, 2),
                    "rmse_raw": round(rmse_raw, 4),
                    "rmse_clean": round(rmse_clean, 4),
                    "rmse_improvement": round((rmse_raw - rmse_clean), 4)
                }
                
        except Exception as e:
            logger.error(f"ML Evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_against_benchmark(self, df_clean: pd.DataFrame, df_benchmark: pd.DataFrame) -> dict:
        """
        Compares the pipeline-cleaned dataset against a human-curated benchmark
        (ground truth) dataset to calculate cleaning accuracy.
        
        Metrics computed:
        - Cell-level accuracy: What % of individual cells match the benchmark exactly
        - Column-level accuracy: Per-column match rate
        - Row-level exact match: What % of rows are identical to benchmark
        - Schema accuracy: Do the columns match the benchmark structure
        - Null accuracy: Did we handle nulls correctly vs benchmark
        - Value accuracy: For non-null cells, do the values match
        """
        report = {
            "overall_cell_accuracy": 0.0,
            "column_accuracy": {},
            "row_exact_match_rate": 0.0,
            "schema_match": {},
            "null_handling_accuracy": 0.0,
            "value_match_accuracy": 0.0,
            "summary": ""
        }
        
        # ── Schema Comparison ───────────────────────────────────────────
        benchmark_cols = set(df_benchmark.columns)
        clean_cols = set(df_clean.columns)
        common_cols = sorted(benchmark_cols & clean_cols)
        missing_cols = sorted(benchmark_cols - clean_cols)
        extra_cols = sorted(clean_cols - benchmark_cols)
        
        schema_precision = len(common_cols) / len(clean_cols) if len(clean_cols) > 0 else 0
        schema_recall = len(common_cols) / len(benchmark_cols) if len(benchmark_cols) > 0 else 0
        schema_f1 = (2 * schema_precision * schema_recall / (schema_precision + schema_recall)
                     if (schema_precision + schema_recall) > 0 else 0)
        
        report["schema_match"] = {
            "benchmark_columns": sorted(benchmark_cols),
            "cleaned_columns": sorted(clean_cols),
            "common_columns": common_cols,
            "missing_from_cleaned": missing_cols,
            "extra_in_cleaned": extra_cols,
            "schema_precision": round(schema_precision, 4),
            "schema_recall": round(schema_recall, 4),
            "schema_f1": round(schema_f1, 4)
        }
        
        if not common_cols:
            report["summary"] = "No common columns between cleaned and benchmark datasets."
            logger.warning(report["summary"])
            return report
        
        # ── Align the DataFrames ────────────────────────────────────────
        # Use only common columns and align row count to the smaller
        min_rows = min(len(df_clean), len(df_benchmark))
        df_c = df_clean[common_cols].head(min_rows).reset_index(drop=True)
        df_b = df_benchmark[common_cols].head(min_rows).reset_index(drop=True)
        
        # Normalize both dataframes for fair comparison
        # Strip whitespace, lowercase strings, convert to string for comparison
        def normalize_cell(val):
            if pd.isna(val):
                return "__NULL__"
            s = str(val).strip().lower()
            # Normalize numeric strings: remove trailing .0
            try:
                f = float(s)
                if f == int(f):
                    return str(int(f))
                return str(round(f, 4))
            except (ValueError, OverflowError):
                return s
        
        df_c_norm = df_c.applymap(normalize_cell)
        df_b_norm = df_b.applymap(normalize_cell)
        
        # ── Cell-Level Accuracy ─────────────────────────────────────────
        total_cells = df_c_norm.shape[0] * df_c_norm.shape[1]
        matching_cells = (df_c_norm == df_b_norm).sum().sum()
        cell_accuracy = matching_cells / total_cells if total_cells > 0 else 0
        report["overall_cell_accuracy"] = round(cell_accuracy * 100, 2)
        
        # ── Per-Column Accuracy ─────────────────────────────────────────
        for col in common_cols:
            col_matches = (df_c_norm[col] == df_b_norm[col]).sum()
            col_total = len(df_c_norm[col])
            col_acc = col_matches / col_total if col_total > 0 else 0
            
            # Also track what kind of mismatches exist
            mismatches = df_c_norm[col] != df_b_norm[col]
            mismatch_count = int(mismatches.sum())
            
            # Sample a few mismatches for debugging
            mismatch_samples = []
            if mismatch_count > 0:
                mismatch_idx = mismatches[mismatches].index[:3]
                for idx in mismatch_idx:
                    mismatch_samples.append({
                        "row": int(idx),
                        "cleaned_value": str(df_c.iloc[idx][col]),
                        "benchmark_value": str(df_b.iloc[idx][col])
                    })
            
            report["column_accuracy"][col] = {
                "accuracy": round(col_acc * 100, 2),
                "matches": int(col_matches),
                "total": col_total,
                "mismatches": mismatch_count,
                "mismatch_samples": mismatch_samples
            }
        
        # ── Row-Level Exact Match ───────────────────────────────────────
        row_matches = (df_c_norm == df_b_norm).all(axis=1).sum()
        row_match_rate = row_matches / min_rows if min_rows > 0 else 0
        report["row_exact_match_rate"] = round(row_match_rate * 100, 2)
        
        # ── Null Handling Accuracy ──────────────────────────────────────
        # How well did we handle nulls? Compare null positions
        benchmark_nulls = (df_b_norm == "__NULL__")
        cleaned_nulls = (df_c_norm == "__NULL__")
        
        # Cells that should be null and are null (true negatives for nulls)
        # Cells that should not be null and are not null (true positives)
        null_correct = ((benchmark_nulls & cleaned_nulls) | (~benchmark_nulls & ~cleaned_nulls)).sum().sum()
        null_accuracy = null_correct / total_cells if total_cells > 0 else 0
        report["null_handling_accuracy"] = round(null_accuracy * 100, 2)
        
        # ── Value Match Accuracy (excluding nulls) ──────────────────────
        # For cells where both have actual values, do they match?
        both_non_null = (~benchmark_nulls & ~cleaned_nulls)
        non_null_total = both_non_null.sum().sum()
        if non_null_total > 0:
            non_null_matches = ((df_c_norm == df_b_norm) & both_non_null).sum().sum()
            value_accuracy = non_null_matches / non_null_total
            report["value_match_accuracy"] = round(value_accuracy * 100, 2)
        else:
            report["value_match_accuracy"] = 100.0
        
        # ── Row Count Comparison ────────────────────────────────────────
        report["row_comparison"] = {
            "cleaned_rows": len(df_clean),
            "benchmark_rows": len(df_benchmark),
            "rows_compared": min_rows,
            "row_count_match": len(df_clean) == len(df_benchmark)
        }
        
        # ── Summary ────────────────────────────────────────────────────
        report["summary"] = (
            f"Benchmark Accuracy: {report['overall_cell_accuracy']}% cell-level match | "
            f"{report['row_exact_match_rate']}% exact row match | "
            f"{report['value_match_accuracy']}% value accuracy (non-null) | "
            f"Schema F1: {report['schema_match']['schema_f1']}"
        )
        
        return report
