import json
import logging
from llm_client import LLMBackend

logger = logging.getLogger(__name__)

class WorkflowGenerator:
    """
    Generates a coherent data cleaning workflow (JSON) based on the profile and detected issues.
    """
    def __init__(self, llm_client: LLMBackend):
        self.llm = llm_client

    def generate_workflow(self, profile: dict, issues: list) -> dict:
        """
        Constructs a prompt to generate the cleaning workflow.
        """
        # Create a condensed context
        context = {
            "columns": profile["columns"],
            "detected_issues": issues
        }
        
        prompt = f"""
        You are a Senior Data Engineer. You have detected the following issues in a dataset:
        {json.dumps(issues, indent=2)}
        
        The dataset details are:
        {json.dumps(context, indent=2)}
        
        YOUR TASK:
        Generate a JSON workflow to clean this data.
        The workflow must follow this exact schema:
        {{
            "steps": [
                {{
                    "step_id": 1,
                    "column": "column_name",
                    "operation": "operation_name",
                    "params": {{ ... }}
                }}
            ]
        }}
        
        SUPPORTED OPERATIONS:
        1. "drop_na": params: {{"threshold": 0.5}} (optional)
        2. "fill_na": params: {{"value": "..."}} or {{"strategy": "mean/median/mode"}}
        3. "drop_duplicates": params: {{}}
        4. "cast_type": params: {{"dtype": "int/float/str/datetime"}}
        5. "replace": params: {{"old": "...", "new": "..."}}
        6. "rename": params: {{"new_name": "..."}}
        7. "remove_outliers": params: {{"method": "z-score", "threshold": 3}}
        
        RULES:
        - Only include steps that fix the identified issues.
        - If "age" has "?" values, use "replace" to change "?" to NaN, then "fill_na" if needed.
        - Ensure step_id is sequential.
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.llm.generate(prompt)
            clean_response = response.replace("```json", "").replace("```", "").strip()
            workflow = json.loads(clean_response)
            return workflow
        except Exception as e:
            logger.error(f"Workflow generation failed: {e}")
            return {"steps": [], "error": str(e)}
