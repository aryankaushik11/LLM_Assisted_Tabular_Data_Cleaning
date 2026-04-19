import pandas as pd
import numpy as np
import re
import logging
import json
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Phase execution order
PHASE_ORDER = ["treat_nulls", "treat_duplicates", "populate_missing", "drop_columns", "split_columns", "standardize"]

# Patterns that indicate time-only values (no date component)
TIME_ONLY_PATTERNS = [
    re.compile(r'^\d{1,2}:\d{2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?|AM|PM)?$', re.IGNORECASE),  # 2:35 p.m., 14:35
    re.compile(r'^\d{1,2}\s*(a\.?m\.?|p\.?m\.?|AM|PM)$', re.IGNORECASE),  # 2 pm, 3 a.m.
]


def is_time_only_column(series: pd.Series) -> bool:
    """
    Checks if a column contains time-only values (no date component).
    Samples non-null values and checks against time-only patterns.
    """
    sample = series.dropna().astype(str).str.strip().head(20)
    if len(sample) == 0:
        return False
    
    matches = 0
    for val in sample:
        if val.lower() in ('nan', 'none', 'nat', ''):
            continue
        for pattern in TIME_ONLY_PATTERNS:
            if pattern.match(val):
                matches += 1
                break
    
    # If at least 60% of non-empty samples match time patterns, it's time-only
    return matches / max(len(sample), 1) > 0.6


def standardize_time_string(val):
    """
    Converts a time string like '2:35 p.m.' to a standardized format '00/00/0000 HH:MM'.
    Returns the original value if parsing fails.
    """
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.lower() in ('nan', 'none', 'nat', ''):
        return np.nan
    try:
        # Extract time part using regex to ignore " (Estimated runway)" etc.
        import re
        time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|AM|PM)?|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?|AM|PM))', s, re.IGNORECASE)
        if time_match:
            s_time = time_match.group(1)
        else:
            s_time = s
            
        # Pre-clean common formats that pd.to_datetime struggles with
        clean_s = s_time.replace('a.m.', 'AM').replace('p.m.', 'PM')
        clean_s = clean_s.replace('A.M.', 'AM').replace('P.M.', 'PM')
        # Prepend a dummy date to avoid OutOfBoundsDatetime for '3 AM'
        parsed = pd.to_datetime("2000-01-01 " + clean_s, errors='raise')
        return parsed.strftime('00/00/0000 %H:%M')
    except Exception:
        # If it happens to be a valid date, parse and format it fully
        try:
            parsed = pd.to_datetime(s, errors='raise')
            return parsed.strftime('%Y-%m-%d %H:%M')
        except Exception:
            return val


class Executor:
    """
    Executes the JSON workflow on the DataFrame.
    Provides detailed per-category output showing what each step did.
    """
    
    def __init__(self):
        self.execution_report = OrderedDict()
        for phase in PHASE_ORDER:
            self.execution_report[phase] = {
                "steps_executed": 0,
                "steps_skipped": 0,
                "details": []
            }
        self.execution_report["other"] = {
            "steps_executed": 0,
            "steps_skipped": 0,
            "details": []
        }
    
    def execute(self, df: pd.DataFrame, workflow: dict) -> pd.DataFrame:
        """
        Applies transformations step-by-step with detailed reporting.
        """
        df_clean = df.copy()
        
        steps = workflow.get("steps", [])
        logger.info(f"Executing {len(steps)} cleaning steps across {len(PHASE_ORDER)} phases.")
        
        # Sort steps by phase order, then by step_id
        phase_priority = {phase: i for i, phase in enumerate(PHASE_ORDER)}
        steps_sorted = sorted(steps, key=lambda s: (
            phase_priority.get(s.get("phase", "other"), 99),
            s.get("step_id", 999)
        ))
        
        for step in steps_sorted:
            try:
                op = step.get("operation")
                col = step.get("column")
                params = step.get("params", {})
                phase = step.get("phase", "other")
                reason = step.get("reason", "")
                step_id = step.get("step_id", "?")
                
                # Track state before execution
                rows_before = len(df_clean)
                cols_before = list(df_clean.columns)
                nulls_before = int(df_clean.isnull().sum().sum()) if col is None else (
                    int(df_clean[col].isnull().sum()) if col and col in df_clean.columns else 0
                )
                
                logger.info(f"Step {step_id} [{phase}]: {op} on '{col}' - {reason}")
                
                # Skip no-op steps
                if op == "no_action":
                    self._record_step(phase, step_id, op, col, "skipped", reason, "No action needed")
                    continue
                
                # ── OPERATION: replace ──────────────────────────────────
                if op == "replace":
                    old_val = params.get("old")
                    new_val = params.get("new")
                    if col and col in df_clean.columns:
                        # Handle "NaN" string -> actual NaN
                        if new_val == "NaN" or new_val == "nan":
                            new_val = np.nan
                        count_replaced = int((df_clean[col].astype(str).str.strip() == str(old_val)).sum())
                        df_clean[col] = df_clean[col].replace(old_val, new_val)
                        # Also try stripped matching
                        if df_clean[col].dtype == object:
                            df_clean[col] = df_clean[col].apply(
                                lambda x: new_val if isinstance(x, str) and x.strip() == str(old_val) else x
                            )
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Replaced '{old_val}' -> '{new_val}' in {count_replaced} cells")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")
                
                # ── OPERATION: fill_na ──────────────────────────────────
                elif op == "fill_na":
                    strategy = params.get("strategy")
                    value = params.get("value")
                    group_by = params.get("group_by") # NEW: Group by support
                    if col and col in df_clean.columns:
                        nulls_in_col = int(df_clean[col].isnull().sum())
                        
                        desc_prefix = ""
                        
                        # Apply group_by if valid
                        if group_by and group_by in df_clean.columns:
                            desc_prefix = f"[Grouped by {group_by}] "
                            
                            def group_impute(x):
                                if strategy == "mean" and np.issubdtype(x.dtype, np.number):
                                    return x.fillna(x.mean())
                                elif strategy == "median" and np.issubdtype(x.dtype, np.number):
                                    return x.fillna(x.median())
                                elif strategy == "mode":
                                    m = x.mode()
                                    return x.fillna(m[0] if len(m) > 0 else np.nan)
                                elif strategy == "ffill":
                                    return x.ffill()
                                elif strategy == "bfill":
                                    return x.bfill()
                                # default fallback within group
                                m = x.mode()
                                return x.fillna(m[0] if len(m) > 0 else np.nan)
                                
                            df_clean[col] = df_clean.groupby(group_by)[col].transform(group_impute)
                            
                            # Fallback if any nulls remain after grouped imputation
                            remaining_nulls = int(df_clean[col].isnull().sum())
                            if remaining_nulls > 0:
                                # Fallback to global
                                if strategy == "mean" and np.issubdtype(df_clean[col].dtype, np.number):
                                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                                elif strategy == "median" and np.issubdtype(df_clean[col].dtype, np.number):
                                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                                elif strategy == "mode":
                                    m = df_clean[col].mode()
                                    df_clean[col] = df_clean[col].fillna(m[0] if len(m) > 0 else value)
                                elif strategy in ["ffill", "bfill", "interpolate"]:
                                    m = df_clean[col].mode()
                                    df_clean[col] = df_clean[col].fillna(m[0] if len(m) > 0 else value)
                            
                            desc = f"{desc_prefix}Filled {nulls_in_col} nulls with {strategy}"
                            
                        else:
                            # Standard global imputation
                            if strategy == "mean" and np.issubdtype(df_clean[col].dtype, np.number):
                                fill_val = df_clean[col].mean()
                                df_clean[col] = df_clean[col].fillna(fill_val)
                                desc = f"Filled {nulls_in_col} nulls with mean={fill_val:.2f}"
                            elif strategy == "median" and np.issubdtype(df_clean[col].dtype, np.number):
                                fill_val = df_clean[col].median()
                                df_clean[col] = df_clean[col].fillna(fill_val)
                                desc = f"Filled {nulls_in_col} nulls with median={fill_val:.2f}"
                            elif strategy == "mode":
                                mode_vals = df_clean[col].mode()
                                fill_val = mode_vals[0] if len(mode_vals) > 0 else value
                                df_clean[col] = df_clean[col].fillna(fill_val)
                                desc = f"Filled {nulls_in_col} nulls with mode='{fill_val}'"
                            elif strategy == "ffill":
                                df_clean[col] = df_clean[col].ffill()
                                desc = f"Forward-filled {nulls_in_col} nulls"
                            elif strategy == "bfill":
                                df_clean[col] = df_clean[col].bfill()
                                desc = f"Back-filled {nulls_in_col} nulls"
                            elif strategy == "interpolate":
                                if np.issubdtype(df_clean[col].dtype, np.number):
                                    df_clean[col] = df_clean[col].interpolate(method='linear')
                                    desc = f"Interpolated {nulls_in_col} nulls (linear)"
                                else:
                                    mode_vals = df_clean[col].mode()
                                    fill_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                                    df_clean[col] = df_clean[col].fillna(fill_val)
                                    desc = f"Non-numeric: filled {nulls_in_col} nulls with mode='{fill_val}'"
                            else:
                                fill_val = value if value is not None else "Unknown"
                                df_clean[col] = df_clean[col].fillna(fill_val)
                                desc = f"Filled {nulls_in_col} nulls with constant='{fill_val}'"
                            
                        self._record_step(phase, step_id, op, col, "executed", reason, desc)
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")
                
                # ── OPERATION: drop_na ──────────────────────────────────
                elif op == "drop_na":
                    if col and col in df_clean.columns:
                        df_clean = df_clean.dropna(subset=[col])
                    else:
                        df_clean = df_clean.dropna()
                    rows_dropped = rows_before - len(df_clean)
                    self._record_step(phase, step_id, op, col, "executed", reason,
                                      f"Dropped {rows_dropped} rows with null values")
                
                # ── OPERATION: drop_duplicates ──────────────────────────
                elif op == "drop_duplicates":
                    subset = params.get("subset")
                    if subset:
                        # Validate subset columns exist
                        valid_subset = [c for c in subset if c in df_clean.columns]
                        if valid_subset:
                            df_clean = df_clean.drop_duplicates(subset=valid_subset)
                        else:
                            df_clean = df_clean.drop_duplicates()
                    else:
                        df_clean = df_clean.drop_duplicates()
                    rows_dropped = rows_before - len(df_clean)
                    self._record_step(phase, step_id, op, col, "executed", reason,
                                      f"Dropped {rows_dropped} duplicate rows")
                    
                # ── OPERATION: drop_column ──────────────────────────────
                elif op == "drop_column":
                    if col and col in df_clean.columns:
                        df_clean = df_clean.drop(columns=[col])
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Dropped column '{col}' ({len(cols_before)} -> {len(df_clean.columns)} columns)")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found - already dropped or doesn't exist")
                
                # ── OPERATION: split_column ─────────────────────────────
                elif op == "split_column":
                    delimiter = params.get("delimiter", ",")
                    new_columns = params.get("new_columns", [])
                    if col and col in df_clean.columns:
                        # Perform the split
                        split_result = df_clean[col].astype(str).str.split(delimiter, expand=True)
                        
                        # Determine new column names
                        n_parts = split_result.shape[1]
                        if new_columns and len(new_columns) >= n_parts:
                            col_names = new_columns[:n_parts]
                        else:
                            col_names = [f"{col}_part{i+1}" for i in range(n_parts)]
                        
                        split_result.columns = col_names
                        # Strip whitespace from split values
                        for c in col_names:
                            split_result[c] = split_result[c].str.strip()
                        
                        # Add new columns and drop original
                        df_clean = pd.concat([df_clean, split_result], axis=1)
                        df_clean = df_clean.drop(columns=[col])
                        
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Split '{col}' into {n_parts} columns: {col_names}")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")
                
                # ── OPERATION: cast_type ────────────────────────────────
                elif op == "cast_type":
                    dtype = params.get("dtype")
                    if col and col in df_clean.columns:
                        if dtype == "int":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
                        elif dtype == "float":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        elif dtype == "datetime":
                            # Check if column is time-only before converting
                            if is_time_only_column(df_clean[col]):
                                df_clean[col] = df_clean[col].apply(standardize_time_string)
                                self._record_step(phase, step_id, op, col, "executed", reason,
                                                  f"Standardized time-only column '{col}' to HH:MM (no date prepended)")
                                continue
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        else:
                            df_clean[col] = df_clean[col].astype(dtype)
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Cast '{col}' to {dtype}")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")

                # ── OPERATION: rename ───────────────────────────────────
                elif op == "rename":
                    new_name = params.get("new_name")
                    if col and col in df_clean.columns and new_name:
                        df_clean = df_clean.rename(columns={col: new_name})
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Renamed '{col}' -> '{new_name}'")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found or no new_name provided")

                # ── OPERATION: remove_outliers ──────────────────────────
                elif op == "remove_outliers":
                    method = params.get("method", "z-score")
                    threshold = params.get("threshold", 3)
                    if col and col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
                        if method == "z-score":
                            mean = df_clean[col].mean()
                            std = df_clean[col].std()
                            if std > 0:
                                z_scores = np.abs((df_clean[col] - mean) / std)
                                df_clean = df_clean[z_scores < threshold]
                        elif method == "iqr":
                            q1 = df_clean[col].quantile(0.25)
                            q3 = df_clean[col].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
                        rows_removed = rows_before - len(df_clean)
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Removed {rows_removed} outliers via {method}")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found or not numeric")
                
                # ── OPERATION: strip_whitespace ─────────────────────────
                elif op == "strip_whitespace":
                    if col == "ALL_OBJECT" or col is None:
                        obj_cols = df_clean.select_dtypes(include=['object']).columns
                        for oc in obj_cols:
                            df_clean[oc] = df_clean[oc].astype(str).str.strip()
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Stripped whitespace from {len(obj_cols)} string columns")
                    elif col and col in df_clean.columns:
                        df_clean[col] = df_clean[col].astype(str).str.strip()
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Stripped whitespace from column '{col}'")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")

                # ── OPERATION: clean_text ───────────────────────────────
                elif op == "clean_text":
                    remove_chars = params.get("remove_chars", "")
                    case = params.get("case", None)
                    if col and col in df_clean.columns:
                        if remove_chars:
                            import re
                            escaped_chars = re.escape(remove_chars)
                            df_clean[col] = df_clean[col].astype(str).str.replace(f"[{escaped_chars}]", "", regex=True)
                        if case == "lower":
                            df_clean[col] = df_clean[col].astype(str).str.lower()
                        elif case == "upper":
                            df_clean[col] = df_clean[col].astype(str).str.upper()
                        elif case == "title":
                            df_clean[col] = df_clean[col].astype(str).str.title()
                        self._record_step(phase, step_id, op, col, "executed", reason,
                                          f"Cleaned text in '{col}' (removed: '{remove_chars}', case: {case})")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")

                # ── OPERATION: format_datetime ──────────────────────────
                elif op == "format_datetime":
                    date_format = params.get("format", "%Y-%m-%d")
                    if col and col in df_clean.columns:
                        # Check if the column is time-only; if so, standardize to HH:MM
                        if is_time_only_column(df_clean[col]):
                            df_clean[col] = df_clean[col].apply(standardize_time_string)
                            self._record_step(phase, step_id, op, col, "executed", reason,
                                              f"Standardized time-only column '{col}' to HH:MM format")
                        else:
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce').dt.strftime(date_format)
                            self._record_step(phase, step_id, op, col, "executed", reason,
                                              f"Formatted datetime in '{col}' to '{date_format}'")
                    else:
                        self._record_step(phase, step_id, op, col, "skipped", reason,
                                          f"Column '{col}' not found")
                
                else:
                    logger.warning(f"Unknown operation: {op}")
                    self._record_step(phase, step_id, op, col, "skipped", reason,
                                      f"Unknown operation '{op}'")
                
            except Exception as e:
                logger.error(f"Failed to execute step {step}: {e}")
                phase = step.get("phase", "other")
                self._record_step(phase, step.get("step_id", "?"), 
                                  step.get("operation", "?"), step.get("column", "?"),
                                  "failed", "", str(e))
                
        return df_clean
    
    def _record_step(self, phase: str, step_id, operation: str, column: str, 
                     status: str, reason: str, detail: str):
        """Records the result of each step for the execution report."""
        if phase not in self.execution_report:
            phase = "other"
        
        entry = {
            "step_id": step_id,
            "operation": operation,
            "column": column,
            "status": status,
            "reason": reason,
            "result": detail
        }
        
        self.execution_report[phase]["details"].append(entry)
        if status == "executed":
            self.execution_report[phase]["steps_executed"] += 1
        else:
            self.execution_report[phase]["steps_skipped"] += 1
    
    def get_execution_report(self) -> dict:
        """Returns the detailed execution report categorized by phase."""
        return dict(self.execution_report)
    
    def print_execution_report(self):
        """Prints a formatted execution report to the logger."""
        report = self.get_execution_report()
        
        phase_labels = {
            "treat_nulls": "PHASE 1 — TREAT NULLS",
            "treat_duplicates": "PHASE 2 — TREAT DUPLICATES",
            "populate_missing": "PHASE 3 — POPULATE MISSING ROWS",
            "drop_columns": "PHASE 4 — DROP UNNEEDED COLUMNS",
            "split_columns": "PHASE 5 — SPLIT COLUMNS",
            "other": "OTHER OPERATIONS"
        }
        
        for phase_key in PHASE_ORDER + ["other"]:
            phase_data = report.get(phase_key, {})
            details = phase_data.get("details", [])
            
            if not details:
                continue
            
            label = phase_labels.get(phase_key, phase_key.upper())
            logger.info(f"\n{'='*60}")
            logger.info(f"  {label}")
            logger.info(f"  Executed: {phase_data['steps_executed']} | Skipped: {phase_data['steps_skipped']}")
            logger.info(f"{'='*60}")
            
            for d in details:
                status_icon = "[OK]" if d["status"] == "executed" else ("[FAIL]" if d["status"] == "failed" else "[SKIP]")
                logger.info(f"  [{status_icon}] Step {d['step_id']}: {d['operation']} -> {d['column']}")
                logger.info(f"      {d['result']}")
