import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# Common placeholder values that should be treated as missing
PLACEHOLDER_VALUES = ["?", "N/A", "n/a", "NA", "na", "NaN", "nan", "None", "none",
                      "--", "-", ".", "..", "...", "missing", "Missing", "MISSING",
                      "undefined", "Undefined", "UNDEFINED", "null", "Null", "NULL",
                      "not available", "Not Available", "unknown", "Unknown", "UNKNOWN",
                      "-99", "-999", "99999", "#N/A", "#NA", "#VALUE!", ""]


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
            "overall_health": {},
            "duplicate_analysis": {},
            "splittable_columns": [],
            "droppable_columns": []
        }
        
        # ── Duplicate Analysis ──────────────────────────────────────────
        duplicate_count = int(df.duplicated().sum())
        profile["duplicate_analysis"] = {
            "total_duplicates": duplicate_count,
            "duplicate_ratio": round(duplicate_count / len(df), 4) if len(df) > 0 else 0,
            "columns_with_high_duplication": []
        }
        
        for col in df.columns:
            col_data = df[col]
            dtype = str(col_data.dtype)
            
            # ── Placeholder Detection ───────────────────────────────────
            placeholder_count = 0
            detected_placeholders = []
            if col_data.dtype == object:
                stripped = col_data.astype(str).str.strip()
                for pv in PLACEHOLDER_VALUES:
                    count = int((stripped == pv).sum())
                    if count > 0:
                        placeholder_count += count
                        detected_placeholders.append({"value": pv, "count": count})
            
            # ── Basic Calculations ──────────────────────────────────────
            # True missing = pandas NaN + placeholder values
            pandas_missing = int(col_data.isnull().sum())
            total_missing = pandas_missing + placeholder_count
            missing_ratio = float(total_missing / len(df)) if len(df) > 0 else 0
            unique_count = int(col_data.nunique())
            
            col_stats = {
                "dtype": dtype,
                "missing_count": total_missing,
                "pandas_null_count": pandas_missing,
                "placeholder_count": placeholder_count,
                "detected_placeholders": detected_placeholders,
                "missing_ratio": round(missing_ratio, 4),
                "unique_count": unique_count,
                "sample_values": col_data.dropna().sample(min(5, len(col_data.dropna())), random_state=42).tolist() if len(col_data.dropna()) > 0 else []
            }

            # ── Type Specific Analysis ──────────────────────────────────
            if np.issubdtype(col_data.dtype, np.number):
                col_stats.update({
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "std": float(col_data.std()) if not col_data.empty else None,
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "median": float(col_data.median()) if not col_data.empty else None,
                    "skewness": float(col_data.skew()) if not col_data.empty else None,
                    "zeros_count": int((col_data == 0).sum()),
                    "q1": float(col_data.quantile(0.25)) if not col_data.empty else None,
                    "q3": float(col_data.quantile(0.75)) if not col_data.empty else None
                })
                # Outlier detection (Z-score > 3)
                if col_stats["std"] and col_stats["std"] > 0:
                    z_scores = np.abs((col_data - col_stats["mean"]) / col_stats["std"])
                    outliers = int((z_scores > 3).sum())
                    col_stats["potential_outliers"] = outliers
                else:
                    col_stats["potential_outliers"] = 0
                    
                # IQR-based outlier detection
                if col_stats.get("q1") is not None and col_stats.get("q3") is not None:
                    iqr = col_stats["q3"] - col_stats["q1"]
                    lower_bound = col_stats["q1"] - 1.5 * iqr
                    upper_bound = col_stats["q3"] + 1.5 * iqr
                    iqr_outliers = int(((col_data < lower_bound) | (col_data > upper_bound)).sum())
                    col_stats["iqr_outliers"] = iqr_outliers

            else:
                # Categorical analysis
                try:
                    top_values = col_data.value_counts().head(10).to_dict()
                    col_stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                except Exception:
                    col_stats["top_values"] = {}
                
                # Semantic check candidates (if low quantity of unique values but object type)
                if unique_count < 50 and unique_count > 0:
                    col_stats["is_categorical"] = True
                    
                # ── Splittable Column Detection ─────────────────────────
                # Check if values contain common delimiters suggesting the column can be split
                if col_data.dtype == object:
                    sample_vals = col_data.dropna().head(100).astype(str)
                    for delimiter in [",", ";", "|", " - ", " / ", ":"]:
                        contains_delim = sample_vals.str.contains(delimiter, na=False, regex=False)
                        delim_ratio = contains_delim.sum() / len(sample_vals) if len(sample_vals) > 0 else 0
                        if delim_ratio > 0.3:  # >30% of values contain the delimiter
                            profile["splittable_columns"].append({
                                "column": col,
                                "delimiter": delimiter,
                                "occurrence_ratio": round(float(delim_ratio), 4)
                            })
                            col_stats["splittable"] = True
                            col_stats["suggested_delimiter"] = delimiter
                            break
            
            # ── Droppable Column Detection ──────────────────────────────
            # Flag columns that are likely unneeded
            is_droppable = False
            drop_reason = []
            
            # Single unique value (constant column)
            if unique_count <= 1:
                is_droppable = True
                drop_reason.append("constant_value")
            
            # >95% missing
            if missing_ratio > 0.95:
                is_droppable = True
                drop_reason.append("almost_all_missing")
            
            # Redundant: if two columns encode the same info (e.g., education & education.num)
            # Flagged during column_analysis pass below
            
            # High cardinality ID-like columns (unique count == row count for non-numeric)
            if col_data.dtype == object and unique_count == len(df) and len(df) > 100:
                is_droppable = True
                drop_reason.append("likely_unique_id")
            
            if is_droppable:
                profile["droppable_columns"].append({
                    "column": col,
                    "reasons": drop_reason
                })
                col_stats["suggested_drop"] = True
                col_stats["drop_reasons"] = drop_reason
            
            # ── Column-Level Duplication ────────────────────────────────
            col_dup_ratio = 1 - (unique_count / len(df)) if len(df) > 0 else 0
            if col_dup_ratio > 0.99 and unique_count > 1:
                profile["duplicate_analysis"]["columns_with_high_duplication"].append({
                    "column": col,
                    "unique_count": unique_count,
                    "duplication_ratio": round(col_dup_ratio, 4)
                })
            
            profile["column_analysis"][col] = col_stats
        
        # ── Redundant Column Pair Detection ─────────────────────────────
        redundant_pairs = self._detect_redundant_pairs(df)
        if redundant_pairs:
            profile["droppable_columns"].extend(redundant_pairs)
        
        # ── Overall Health Score ────────────────────────────────────────
        total_cells = len(df) * len(df.columns)
        total_missing = sum(v["missing_count"] for v in profile["column_analysis"].values())
        total_placeholders = sum(v["placeholder_count"] for v in profile["column_analysis"].values())
        
        profile["overall_health"] = {
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "total_placeholder_cells": total_placeholders,
            "completeness_score": round(1 - (total_missing / total_cells), 4) if total_cells > 0 else 0,
            "duplicate_rows": duplicate_count,
            "droppable_columns_count": len(profile["droppable_columns"]),
            "splittable_columns_count": len(profile["splittable_columns"])
        }

        return profile
    
    def _detect_redundant_pairs(self, df: pd.DataFrame) -> list:
        """
        Detects pairs of columns where one is a numeric encoding of the other.
        E.g., 'education' and 'education.num' in the Adult dataset.
        """
        redundant = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for ncol in numeric_cols:
            for ocol in object_cols:
                # Check if the column names suggest a relationship
                ncol_base = ncol.replace(".num", "").replace("_num", "").replace("_id", "").replace(".id", "")
                if ncol_base == ocol or ocol.startswith(ncol_base):
                    # Verify: check if there's a 1-to-1 mapping
                    try:
                        combined = df[[ocol, ncol]].dropna()
                        mappings = combined.groupby(ocol)[ncol].nunique()
                        if (mappings == 1).all():
                            redundant.append({
                                "column": ncol,
                                "reasons": [f"redundant_encoding_of_{ocol}"],
                                "related_column": ocol
                            })
                    except Exception:
                        pass
        return redundant

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

    def generate_data_snapshot(self, df: pd.DataFrame, profile: dict, target_col: str = None) -> str:
        """
        Generates a rich, human-readable snapshot of the dataset for LLM context.
        This gives the LLM actual data rows, not just statistics, so it can
        understand real patterns and make accurate cleaning decisions.
        """
        lines = []
        
        # ── Dataset Overview ────────────────────────────────────────────
        lines.append(f"DATASET OVERVIEW:")
        lines.append(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        if target_col:
            lines.append(f"  Target column for analysis: '{target_col}'")
            if target_col in df.columns:
                target_dist = df[target_col].value_counts().head(5).to_dict()
                lines.append(f"  Target distribution: {target_dist}")
        lines.append("")
        
        # ── First 5 rows as a table ─────────────────────────────────────
        lines.append("SAMPLE DATA (first 5 rows):")
        # Use to_string for clean tabular formatting, truncate wide values
        sample_df = df.head(5).copy()
        for col in sample_df.select_dtypes(include=['object']).columns:
            sample_df[col] = sample_df[col].astype(str).str[:30]
        lines.append(sample_df.to_string(index=False))
        lines.append("")
        
        # ── Column-by-column summary ────────────────────────────────────
        lines.append("COLUMN DETAILS:")
        for col in df.columns:
            stats = profile.get("column_analysis", {}).get(col, {})
            dtype = stats.get("dtype", str(df[col].dtype))
            missing = stats.get("missing_count", 0)
            unique = stats.get("unique_count", 0)
            
            col_line = f"  [{col}] type={dtype}, unique={unique}, missing={missing}"
            
            # Add type-specific context
            if np.issubdtype(df[col].dtype, np.number):
                skew = stats.get("skewness", 0)
                mean_val = stats.get("mean", 0)
                median_val = stats.get("median", 0)
                std_val = stats.get("std", 0)
                outliers = stats.get("potential_outliers", 0)
                col_line += f", mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}, skewness={skew:.2f}, outliers={outliers}"
            else:
                top = stats.get("top_values", {})
                if top:
                    top_3 = dict(list(top.items())[:3])
                    col_line += f", top_values={top_3}"
                
                # Show detected placeholders
                phs = stats.get("detected_placeholders", [])
                if phs:
                    ph_str = ", ".join([f"'{p['value']}'({p['count']})" for p in phs])
                    col_line += f", PLACEHOLDERS_FOUND=[{ph_str}]"
            
            lines.append(col_line)
        
        lines.append("")
        
        # ── Cross-column observations ───────────────────────────────────
        droppable = profile.get("droppable_columns", [])
        if droppable:
            lines.append("REDUNDANT/DROPPABLE COLUMNS DETECTED:")
            for d in droppable:
                related = d.get("related_column", "")
                reasons = ", ".join(d.get("reasons", []))
                extra = f" (related to '{related}')" if related else ""
                lines.append(f"  - '{d['column']}': {reasons}{extra}")
            lines.append("")
        
        dup_info = profile.get("duplicate_analysis", {})
        if dup_info.get("total_duplicates", 0) > 0:
            lines.append(f"DUPLICATE ROWS: {dup_info['total_duplicates']} exact duplicates found")
            lines.append("")
        
        splittable = profile.get("splittable_columns", [])
        if splittable:
            lines.append("SPLITTABLE COLUMNS:")
            for s in splittable:
                lines.append(f"  - '{s['column']}': delimiter='{s['delimiter']}', occurrence={s['occurrence_ratio']:.0%}")
            lines.append("")
        
        return "\n".join(lines)
