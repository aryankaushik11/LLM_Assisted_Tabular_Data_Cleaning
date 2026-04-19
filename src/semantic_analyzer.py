import json
import logging
from llm_client import LLMBackend

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Uses an LLM to analyze the Data Profile and detect semantic errors.
    """
    def __init__(self, llm_client: LLMBackend):
        self.llm = llm_client

    def analyze_profile(self, profile: dict, df=None, target_col: str = None) -> dict:
        """
        Sends the profile and real data context to the LLM for issue detection.
        
        Args:
            profile: The statistical profile from DataProfiler.analyze()
            df: The raw DataFrame (optional but recommended for richer context)
            target_col: The target/label column to preserve during cleaning
        """
        # Build the data snapshot if we have the raw dataframe
        data_snapshot = ""
        if df is not None:
            from profiling import DataProfiler
            data_snapshot = DataProfiler().generate_data_snapshot(df, profile, target_col)
        
        # Build compact column stats for the JSON portion
        compact_stats = {}
        for k, v in profile.get("column_analysis", {}).items():
            entry = {
                "dtype": v["dtype"],
                "unique_count": v["unique_count"],
                "missing_count": v.get("missing_count", 0),
                "placeholder_count": v.get("placeholder_count", 0),
                "detected_placeholders": v.get("detected_placeholders", []),
                "top_values": v.get("top_values", {}),
            }
            # Include numeric stats when available
            if v.get("mean") is not None:
                entry["mean"] = round(v["mean"], 2)
                entry["median"] = round(v.get("median", 0), 2)
                entry["std"] = round(v.get("std", 0), 2)
                entry["skewness"] = round(v.get("skewness", 0), 2)
                entry["potential_outliers"] = v.get("potential_outliers", 0)
                entry["zeros_count"] = v.get("zeros_count", 0)
            compact_stats[k] = entry
        
        prompt = f"""You are a Data Quality Expert. You must analyze the dataset below and identify every data quality issue that needs fixing before this data can be used for machine learning.

{data_snapshot}
COLUMN STATISTICS (JSON):
{json.dumps(compact_stats, indent=2)}

DUPLICATE INFO: {json.dumps(profile.get('duplicate_analysis', {}), indent=2, default=str)}
DROPPABLE COLUMNS: {json.dumps(profile.get('droppable_columns', []), indent=2)}
SPLITTABLE COLUMNS: {json.dumps(profile.get('splittable_columns', []), indent=2)}

INSTRUCTIONS:
Look at the SAMPLE DATA rows above carefully. Cross-reference what you see in the actual data with the statistics. For each column, ask yourself:
1. Are there placeholder values (like "?", "N/A", "-99") hiding as valid data? These must be replaced with NaN first.
2. Is this column's dtype correct? (e.g., a numeric column stored as object because of placeholders)
3. Are there inconsistent categorical values? (e.g., leading/trailing spaces, mixed case)
4. Are there duplicate rows that should be removed?
5. Are any columns redundant or unneeded? (e.g., two columns encoding the same info)
6. Are any columns genuinely splittable? (e.g., "City, State" in one column). DO NOT split currency strings or integers (e.g. $780,000)! Instead, those should be cleaned and cast.
7. For numeric columns, does the skewness or outlier count suggest cleaning is needed?
{"8. IMPORTANT: The target column '" + target_col + "' is the label for ML. Do NOT drop it or alter its values." if target_col else ""}
9. Are there data types that should be cast, or text fields that need formatting (e.g. lowering case, stripping $ or commas)?
10. For missing data, are there logical entities (like "flight" or "user_id" or "item_name") that can be grouped to provide accurate contextual imputation instead of global averaging?
11. CRITICAL: Removing/dropping rows should be your absolute LAST option. It hurts downstream benchmark metrics. Always prefer inference or imputation.

OUTPUT FORMAT:
Return a JSON object with a list of "issues". Each issue must have:
- "column": column name (or "ALL" for row-level issues like duplicates)
- "issue_type": one of (null_treatment, placeholder, duplicate, missing_population, drop_column, split_column, formatting, casting, numeric_cleaning, other)
- "category": one of (treat_nulls, treat_duplicates, populate_missing, drop_columns, split_columns, standardize)
- "description": what the problem is, referencing what you see in the actual data
- "severity": "high", "medium", or "low"
- "recommended_action": the specific fix (e.g., "clean_text to remove commas then cast_type to int")

Return ONLY valid JSON. No markdown, no explanation outside the JSON."""
        
        system_instruction = "You are a data quality expert. You analyze real tabular data and return structured JSON reports identifying every data quality issue. Be thorough but precise."
        
        try:
            response = self.llm.generate(prompt, system_instruction)
            # Basic cleanup of code blocks if LLM adds them
            clean_response = response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_response)
            
            # Validate structure
            if "issues" not in parsed:
                if isinstance(parsed, list):
                    parsed = {"issues": parsed}
                else:
                    parsed = {"issues": []}
            
            # Enrich with category counts
            categories = {}
            for issue in parsed["issues"]:
                cat = issue.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            parsed["category_summary"] = categories
            
            return parsed
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON.")
            logger.debug(f"Raw Response: {response}")
            # Attempt best-effort extraction
            return self._fallback_analysis(profile)
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return self._fallback_analysis(profile)
    
    def _fallback_analysis(self, profile: dict) -> dict:
        """
        Rule-based fallback when LLM fails to produce valid JSON.
        Generates issues deterministically from the profile data.
        """
        logger.info("Running fallback rule-based analysis...")
        issues = []
        
        for col, stats in profile.get("column_analysis", {}).items():
            # ── Category 1: Treat Nulls ─────────────────────────────────
            if stats.get("pandas_null_count", 0) > 0:
                missing_ratio = stats.get("missing_ratio", 0)
                if missing_ratio > 0.5:
                    action = "drop_column (>50% missing)"
                    severity = "high"
                elif stats["dtype"] in ("int64", "float64"):
                    # Adapt strategy based on distribution shape
                    skewness = stats.get("skewness", 0)
                    outliers = stats.get("potential_outliers", 0)
                    if abs(skewness) > 1.0 or outliers > 0:
                        action = f"fill_na with median (skewness={skewness:.2f}, outliers={outliers})"
                    else:
                        action = f"fill_na with mean (symmetric distribution, skewness={skewness:.2f})"
                    severity = "medium"
                else:
                    action = "fill_na with mode (categorical column)"
                    severity = "medium"
                    
                issues.append({
                    "column": col,
                    "issue_type": "null_treatment",
                    "category": "treat_nulls",
                    "description": f"Column has {stats['pandas_null_count']} null values ({missing_ratio:.1%} of data)",
                    "severity": severity,
                    "recommended_action": action
                })
            
            # ── Placeholder Detection ───────────────────────────────────
            for ph in stats.get("detected_placeholders", []):
                issues.append({
                    "column": col,
                    "issue_type": "placeholder",
                    "category": "treat_nulls",
                    "description": f"Placeholder '{ph['value']}' found {ph['count']} times, masking true missing data",
                    "severity": "high",
                    "recommended_action": f"replace '{ph['value']}' with NaN, then fill_na with mode"
                })
            
            # ── Category 4: Drop Columns ────────────────────────────────
            if stats.get("suggested_drop", False):
                for reason in stats.get("drop_reasons", []):
                    issues.append({
                        "column": col,
                        "issue_type": "drop_column",
                        "category": "drop_columns",
                        "description": f"Column suggested for dropping: {reason}",
                        "severity": "medium",
                        "recommended_action": "drop_column"
                    })
            
            # ── Category 5: Split Columns ───────────────────────────────
            if stats.get("splittable", False):
                delim = stats.get("suggested_delimiter", ",")
                issues.append({
                    "column": col,
                    "issue_type": "split_column",
                    "category": "split_columns",
                    "description": f"Column contains delimiter '{delim}' and may encode multiple values",
                    "severity": "low",
                    "recommended_action": f"split_column on '{delim}'"
                })
        
        # ── Category 2: Duplicates ──────────────────────────────────────
        dup_info = profile.get("duplicate_analysis", {})
        if dup_info.get("total_duplicates", 0) > 0:
            issues.append({
                "column": "ALL",
                "issue_type": "duplicate",
                "category": "treat_duplicates",
                "description": f"Dataset has {dup_info['total_duplicates']} exact duplicate rows",
                "severity": "high",
                "recommended_action": "drop_duplicates"
            })
        
        # ── Droppable columns from profile ──────────────────────────────
        for drop_info in profile.get("droppable_columns", []):
            col_name = drop_info.get("column", "")
            reasons = drop_info.get("reasons", [])
            # Avoid duplicating issues already added above
            existing_cols = [i["column"] for i in issues if i["issue_type"] == "drop_column"]
            if col_name not in existing_cols:
                issues.append({
                    "column": col_name,
                    "issue_type": "drop_column",
                    "category": "drop_columns",
                    "description": f"Column suggested for dropping: {', '.join(reasons)}",
                    "severity": "medium",
                    "recommended_action": "drop_column"
                })
        
        # Build category summary
        categories = {}
        for issue in issues:
            cat = issue.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "issues": issues,
            "category_summary": categories,
            "source": "fallback_rule_based"
        }
