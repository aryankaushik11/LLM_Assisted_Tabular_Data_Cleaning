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

    def analyze_profile(self, profile: dict) -> dict:
        """
        Sends the profile to the LLM and asks for a structured error report.
        """
        # Compact the profile to save tokens
        compact_profile = {
            "columns": profile["columns"],
            "column_analysis": {
                k: {
                    "dtype": v["dtype"],
                    "sample": v["sample_values"],
                    "unique_count": v["unique_count"],
                    "top_values": v.get("top_values", {}),
                    "potential_outliers": v.get("potential_outliers", 0)
                } 
                for k, v in profile["column_analysis"].items()
            }
        }
        
        prompt = f"""
        You are a Data Quality Expert. Analyze the following JSON data profile of a tabular dataset.
        
        DATA PROFILE:
        {json.dumps(compact_profile, indent=2)}
        
        YOUR TASK:
        Identify potential semantic data quality issues. Focus on:
        1. Inconsistent categorical values (e.g., 'USA' vs 'usa' vs 'U.S.A.').
        2. Numerical columns that shouldn't be numeric (e.g., IDs).
        3. Object columns that should be numeric (e.g., "Age" as string because of "twenty").
        4. Placeholder values (e.g., "?", "N/A", "-99").
        5. Outliers that look suspicious based on the column name (e.g., Age 200).
        
        OUTPUT FORMAT:
        Return a JSON object with a list of "issues". Each issue must have:
        - "column": name of the column
        - "issue_type": type of error (inconsistency, wrong_type, placeholder, outlier)
        - "description": brief explanation
        - "severity": "high" or "low"
        
        Return ONLY valid JSON.
        """
        
        system_instruction = "You are a helpful AI assistant that analyzes tabular data profiles."
        
        try:
            response = self.llm.generate(prompt, system_instruction)
            # Basic cleanup of code blocks if Gemma adds them
            clean_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON.")
            logger.debug(f"Raw Response: {response}")
            return {"error": "Failed to parse analysis", "raw": response}
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {"error": str(e)}
