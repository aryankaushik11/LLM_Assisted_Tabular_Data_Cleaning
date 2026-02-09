import os
import sys
import json
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from ingestion import DataLoader
from profiling import DataProfiler
from llm_client import LLMFactory
from semantic_analyzer import SemanticAnalyzer
from workflow_generator import WorkflowGenerator
from executor import Executor
from evaluation import Evaluator

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="LLM-Assisted Data Cleaning Pipeline")
    parser.add_argument("--dataset", type=str, default="adult.csv", help="Filename in data/raw")
    parser.add_argument("--llm_provider", type=str, default="ollama", help="ollama or google")
    parser.add_argument("--target_col", type=str, default="income", help="Target column for ML evaluation")
    args = parser.parse_args()

    try:
        # 1. Ingestion
        logger.info("=== STEP 1: Data Ingestion ===")
        loader = DataLoader()
        df_raw = loader.load_csv(args.dataset)
        
        # 2. Profiling
        logger.info("=== STEP 2: Data Profiling ===")
        profiler = DataProfiler()
        profile = profiler.analyze(df_raw)
        
        # Save Profile
        with open(f"data/processed/profile_{args.dataset.split('.')[0]}.json", "w") as f:
            json.dump(profile, f, indent=2, default=str)
            
        # 3. LLM Analysis
        logger.info(f"=== STEP 3: Semantic Analysis with {args.llm_provider} ===")
        llm = LLMFactory.create_client(provider=args.llm_provider)
        analyzer = SemanticAnalyzer(llm)
        
        logger.info("Analyzing profile for issues...")
        issues_report = analyzer.analyze_profile(profile)
        
        if "issues" not in issues_report:
            # Fallback if LLM output structure is messy
            logger.warning("LLM report format issue, attempting to extract 'issues' list.")
            issues = issues_report if isinstance(issues_report, list) else []
        else:
            issues = issues_report["issues"]
            
        logger.info(f"Detected {len(issues)} issues.")
        with open("data/processed/issues_report.json", "w") as f:
            json.dump(issues, f, indent=2)

        # 4. Workflow Generation
        logger.info("=== STEP 4: Generating Cleaning Workflow ===")
        wf_gen = WorkflowGenerator(llm)
        workflow = wf_gen.generate_workflow(profile, issues)
        
        logger.info("Generated Workflow:")
        logger.info(json.dumps(workflow, indent=2))
        
        with open("data/processed/cleaning_workflow.json", "w") as f:
            json.dump(workflow, f, indent=2)

        # 5. Execution
        logger.info("=== STEP 5: Executing Workflow ===")
        executor = Executor()
        df_clean = executor.execute(df_raw, workflow)
        
        output_path = f"data/processed/clean_{args.dataset}"
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")

        # 6. Evaluation
        logger.info("=== STEP 6: Evaluation ===")
        evaluator = Evaluator()
        
        quality_metrics = evaluator.evaluate_quality(df_raw, df_clean)
        ml_metrics = evaluator.evaluate_ml_performance(df_raw, df_clean, target_col=args.target_col)
        
        report = {
            "quality_metrics": quality_metrics,
            "ml_metrics": ml_metrics
        }
        
        logger.info("Evaluation Report:")
        logger.info(json.dumps(report, indent=2))
        
        with open("data/processed/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info("=== Pipeline Completed Successfully ===")
        
    except Exception as e:
        logger.exception("Pipeline failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
