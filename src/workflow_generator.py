import json
import logging
from llm_client import LLMBackend

logger = logging.getLogger(__name__)


class WorkflowGenerator:
    """
    Generates a coherent data cleaning workflow (JSON) based on the profile and detected issues.
    Covers all 5 cleaning categories with proper step sequencing.
    """
    def __init__(self, llm_client: LLMBackend):
        self.llm = llm_client

    def generate_workflow(self, profile: dict, issues: list, df=None, target_col: str = None) -> dict:
        """
        Constructs a prompt with full dataset context to generate the cleaning workflow.
        
        Args:
            profile: The statistical profile from DataProfiler.analyze()
            issues: List of detected issues from SemanticAnalyzer
            df: The raw DataFrame (optional but recommended for richer context)
            target_col: The target/label column that must be preserved
        """
        # Build data snapshot for LLM context
        data_snapshot = ""
        if df is not None:
            from profiling import DataProfiler
            data_snapshot = DataProfiler().generate_data_snapshot(df, profile, target_col)
        
        # Build compact column stats so the LLM knows skewness, distributions, etc.
        column_stats = {}
        for k, v in profile.get("column_analysis", {}).items():
            entry = {"dtype": v["dtype"], "missing": v.get("missing_count", 0)}
            if v.get("mean") is not None:
                entry["mean"] = round(v["mean"], 2)
                entry["median"] = round(v.get("median", 0), 2)
                entry["skewness"] = round(v.get("skewness", 0), 2)
                entry["outliers"] = v.get("potential_outliers", 0)
            if v.get("detected_placeholders"):
                entry["placeholders"] = v["detected_placeholders"]
            column_stats[k] = entry
        
        prompt = f"""You are a Senior Data Engineer. Based on the dataset context and detected issues below, generate a precise JSON cleaning workflow.

{data_snapshot}
COLUMN STATISTICS:
{json.dumps(column_stats, indent=2)}

DETECTED ISSUES:
{json.dumps(issues, indent=2)}

DROPPABLE COLUMNS: {json.dumps(profile.get('droppable_columns', []), indent=2)}
SPLITTABLE COLUMNS: {json.dumps(profile.get('splittable_columns', []), indent=2)}

YOUR TASK:
YOUR TASK:
Generate a JSON workflow to clean this data. The workflow MUST address ALL 6 cleaning categories in this EXACT ORDER:

PHASE 1 - TREAT NULLS (phase: "treat_nulls"):
  First replace placeholders (?, N/A, etc.) with NaN, then fill nulls using the appropriate strategy.
  
PHASE 2 - TREAT DUPLICATES (phase: "treat_duplicates"):
  Remove exact duplicate rows.
  
PHASE 3 - POPULATE MISSING ROWS (phase: "populate_missing"):
  Use interpolation for ordered numeric data, or imputation for remaining gaps.
  
PHASE 4 - DROP UNNEEDED COLUMNS (phase: "drop_columns"):
  Drop constant, near-empty, redundant, or ID-like columns.
  
PHASE 5 - SPLIT COLUMNS (phase: "split_columns"):
  Split delimited columns into multiple new columns (e.g. City/State). DO NOT split numbers/currencies!
  
PHASE 6 - STANDARDIZE & FORMAT (phase: "standardize"):
  Clean text (remove currency symbols $, commas ,), format dates (YYYY-MM-DD), and fix data types using cast_type.

SCHEMA:
{{
    "steps": [
        {{
            "step_id": 1,
            "phase": "treat_nulls",
            "column": "column_name",
            "operation": "operation_name",
            "params": {{ ... }},
            "reason": "brief explanation"
        }}
    ]
}}

SUPPORTED OPERATIONS:
1.  "replace"          - params: {{"old": "...", "new": "..."}}
2.  "fill_na"          - params: {{"strategy": "mean/median/mode/ffill/bfill/interpolate", "group_by": "col_name"}} ("group_by" is optional but highly recommended to impute contextually)
3.  "drop_na"          - params: {{}} (drops rows where column is null)
4.  "drop_duplicates"  - params: {{"subset": ["col1","col2"]}} (optional)
5.  "drop_column"      - params: {{}}
6.  "split_column"     - params: {{"delimiter": ",", "new_columns": ["col_a", "col_b"]}}
7.  "cast_type"        - params: {{"dtype": "int/float/str/datetime"}}
8.  "rename"           - params: {{"new_name": "..."}}
9.  "remove_outliers"  - params: {{"method": "z-score/iqr", "threshold": 3}}
10. "strip_whitespace" - params: {{}}
11. "clean_text"       - params: {{"remove_chars": "$,", "case": "title/upper/lower"}} (Removes characters via regex & sets case)
12. "format_datetime"  - params: {{"format": "%Y-%m-%d"}}

RULES:
- Every phase MUST have at least one step. If no issue exists for a phase, add a no_action step with operation "no_action".
- DO NOT use split_column on currencies or numbers (like $780,000,000). Instead, use "clean_text" to remove '$,' and then "cast_type" to "int" or "float" in PHASE 6.
- For columns that contain ONLY time values (like "2:35 p.m.", "14:30"), do NOT convert them to full datetime. The system will standardize them to HH:MM format automatically. Only use format_datetime on columns that actually contain dates.
- If a column has placeholder "?" values, first "replace" them with NaN, then "fill_na".
- Step IDs must be sequential starting from 1.
- Each step MUST have "phase", "column", "operation", "params", and "reason" fields.
{"- CRITICAL: Do NOT drop or alter the target column '" + target_col + "'. It is the ML label." if target_col else ""}

IMPUTATION RULES:
- For "fill_na", always use {{"strategy": "mean/median/mode/ffill/bfill"}}, NOT {{"value": "some_literal"}}.
- If multiple rows belong to the same logical entity (e.g. same 'flight' number, 'user_id', or 'transaction_id'), you MUST use the "group_by" parameter in "fill_na" (e.g., {{"strategy": "mode", "group_by": "flight"}}). This ensures contextual imputation instead of blind global averaging.
- For NUMERIC columns: use "median" if skewness > 1 or outliers > 0; use "mean" if symmetric.
- For CATEGORICAL columns: always use "mode" (preferably with "group_by" if an entity column exists).

Return ONLY valid JSON."""
        
        try:
            response = self.llm.generate(prompt)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            workflow = json.loads(clean_response)
            
            # Validate and enrich
            workflow = self._validate_workflow(workflow, profile, issues)
            return workflow
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM workflow response: {e}")
            logger.info("Generating fallback workflow from detected issues...")
            return self._generate_fallback_workflow(profile, issues)
        except Exception as e:
            logger.error(f"Workflow generation failed: {e}")
            return self._generate_fallback_workflow(profile, issues)
    
    def _validate_workflow(self, workflow: dict, profile: dict, issues: list) -> dict:
        """
        Validates and enriches the LLM-generated workflow.
        Ensures all 5 phases are represented.
        """
        steps = workflow.get("steps", [])
        present_phases = set(s.get("phase", "") for s in steps)
        required_phases = ["treat_nulls", "treat_duplicates", "populate_missing", "drop_columns", "split_columns", "standardize"]
        
        max_id = max((s.get("step_id", 0) for s in steps), default=0)
        
        for phase in required_phases:
            if phase not in present_phases:
                max_id += 1
                steps.append({
                    "step_id": max_id,
                    "phase": phase,
                    "column": None,
                    "operation": "no_action",
                    "params": {},
                    "reason": f"No issues detected for {phase} phase"
                })
        
        workflow["steps"] = steps
        return workflow
    
    def _generate_fallback_workflow(self, profile: dict, issues: list) -> dict:
        """
        Deterministic fallback workflow generation when LLM fails.
        Builds steps directly from detected issues.
        """
        steps = []
        step_id = 0
        
        # ── PHASE 1: Treat Nulls ────────────────────────────────────────
        # First handle placeholders
        for col, stats in profile.get("column_analysis", {}).items():
            for ph in stats.get("detected_placeholders", []):
                step_id += 1
                steps.append({
                    "step_id": step_id,
                    "phase": "treat_nulls",
                    "column": col,
                    "operation": "replace",
                    "params": {"old": ph["value"], "new": "NaN"},
                    "reason": f"Replace placeholder '{ph['value']}' ({ph['count']} occurrences) with NaN"
                })
        
        # Then fill NaN values
        for col, stats in profile.get("column_analysis", {}).items():
            total_missing = stats.get("missing_count", 0)
            if total_missing > 0:
                missing_ratio = stats.get("missing_ratio", 0)
                if missing_ratio > 0.5:
                    # >50% missing: drop the rows — too sparse to impute reliably
                    step_id += 1
                    steps.append({
                        "step_id": step_id,
                        "phase": "treat_nulls",
                        "column": col,
                        "operation": "drop_na",
                        "params": {},
                        "reason": f"Column has {missing_ratio:.0%} missing - too sparse to impute, dropping rows"
                    })
                elif stats["dtype"] in ("int64", "float64"):
                    # Numeric column: choose strategy based on distribution shape
                    skewness = stats.get("skewness", 0)
                    outlier_count = stats.get("potential_outliers", 0)
                    zeros_ratio = stats.get("zeros_count", 0) / max(1, profile.get("rows", 1))
                    
                    if abs(skewness) > 1.0 or outlier_count > 0:
                        # Skewed distribution or outliers present -> median is more robust
                        strategy = "median"
                        reason = (f"Numeric column with {total_missing} missing values - "
                                  f"skewness={skewness:.2f}, outliers={outlier_count}: "
                                  f"using median (robust to skew/outliers)")
                    elif zeros_ratio > 0.5:
                        # Dominated by zeros (e.g., capital.gain) -> median captures the zero-heavy nature
                        strategy = "median"
                        reason = (f"Numeric column with {total_missing} missing values - "
                                  f"{zeros_ratio:.0%} zeros: using median (reflects true center)")
                    else:
                        # Symmetric distribution with no outliers -> mean is appropriate
                        strategy = "mean"
                        reason = (f"Numeric column with {total_missing} missing values - "
                                  f"skewness={skewness:.2f}, no outliers: "
                                  f"using mean (distribution is symmetric)")
                    
                    step_id += 1
                    steps.append({
                        "step_id": step_id,
                        "phase": "treat_nulls",
                        "column": col,
                        "operation": "fill_na",
                        "params": {"strategy": strategy},
                        "reason": reason
                    })
                else:
                    # Categorical column: mode is always the right choice
                    step_id += 1
                    steps.append({
                        "step_id": step_id,
                        "phase": "treat_nulls",
                        "column": col,
                        "operation": "fill_na",
                        "params": {"strategy": "mode"},
                        "reason": f"Categorical column with {total_missing} missing values - filling with mode (most frequent value)"
                    })
        
        if not any(s["phase"] == "treat_nulls" for s in steps):
            step_id += 1
            steps.append({
                "step_id": step_id,
                "phase": "treat_nulls",
                "column": None,
                "operation": "no_action",
                "params": {},
                "reason": "No null treatment needed"
            })
        
        # ── PHASE 2: Treat Duplicates ───────────────────────────────────
        dup_count = profile.get("duplicate_analysis", {}).get("total_duplicates", 0)
        step_id += 1
        if dup_count > 0:
            steps.append({
                "step_id": step_id,
                "phase": "treat_duplicates",
                "column": None,
                "operation": "drop_duplicates",
                "params": {},
                "reason": f"Removing {dup_count} exact duplicate rows"
            })
        else:
            steps.append({
                "step_id": step_id,
                "phase": "treat_duplicates",
                "column": None,
                "operation": "no_action",
                "params": {},
                "reason": "No duplicate rows found"
            })
        
        # ── PHASE 3: Populate Missing Rows ──────────────────────────────
        # Check for numeric columns that might benefit from interpolation
        has_populate_step = False
        for col, stats in profile.get("column_analysis", {}).items():
            if stats["dtype"] in ("int64", "float64") and stats.get("pandas_null_count", 0) > 0:
                # For ordered numeric data, suggest interpolation
                step_id += 1
                steps.append({
                    "step_id": step_id,
                    "phase": "populate_missing",
                    "column": col,
                    "operation": "fill_na",
                    "params": {"strategy": "interpolate"},
                    "reason": f"Interpolate remaining missing values in numeric column '{col}'"
                })
                has_populate_step = True
        
        if not has_populate_step:
            step_id += 1
            steps.append({
                "step_id": step_id,
                "phase": "populate_missing",
                "column": None,
                "operation": "no_action",
                "params": {},
                "reason": "No additional missing row population needed after null treatment"
            })
        
        # ── PHASE 4: Drop Unneeded Columns ──────────────────────────────
        droppable = profile.get("droppable_columns", [])
        if droppable:
            for drop_info in droppable:
                step_id += 1
                steps.append({
                    "step_id": step_id,
                    "phase": "drop_columns",
                    "column": drop_info["column"],
                    "operation": "drop_column",
                    "params": {},
                    "reason": f"Dropping column: {', '.join(drop_info.get('reasons', ['unneeded']))}"
                })
        else:
            step_id += 1
            steps.append({
                "step_id": step_id,
                "phase": "drop_columns",
                "column": None,
                "operation": "no_action",
                "params": {},
                "reason": "No columns identified for dropping"
            })
        
        # ── PHASE 5: Split Columns ──────────────────────────────────────
        splittable = profile.get("splittable_columns", [])
        if splittable:
            for split_info in splittable:
                step_id += 1
                steps.append({
                    "step_id": step_id,
                    "phase": "split_columns",
                    "column": split_info["column"],
                    "operation": "split_column",
                    "params": {
                        "delimiter": split_info["delimiter"],
                        "new_columns": [f"{split_info['column']}_part1", f"{split_info['column']}_part2"]
                    },
                    "reason": f"Split column on delimiter '{split_info['delimiter']}'"
                })
        else:
            step_id += 1
            steps.append({
                "step_id": step_id,
                "phase": "split_columns",
                "column": None,
                "operation": "no_action",
                "params": {},
                "reason": "No columns identified for splitting"
            })
        
        # ── Strip whitespace on all object columns ──────────────────────
        step_id += 1
        steps.append({
            "step_id": step_id,
            "phase": "treat_nulls",
            "column": "ALL_OBJECT",
            "operation": "strip_whitespace",
            "params": {},
            "reason": "Strip leading/trailing whitespace from all string columns"
        })
        
        return {"steps": steps}
