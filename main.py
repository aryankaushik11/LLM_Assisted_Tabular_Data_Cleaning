import os
import sys
import json
import logging
import argparse
import pandas as pd
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
    parser.add_argument("--dataset", type=str, default=None, help="Filename in data/raw (auto-detected if None)")
    parser.add_argument("--llm_provider", type=str, default="ollama", help="ollama or google")
    parser.add_argument("--target_col", type=str, default=None, help="Target column for ML eval (auto-detected if None)")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Auto-detect dataset if not provided
    if args.dataset is None:
        import glob
        csv_files = glob.glob("data/raw/*.csv")
        if not csv_files:
            logger.error("No CSV files found in data/raw directory. Please add one.")
            return
        args.dataset = os.path.basename(csv_files[0])
        logger.info(f"Auto-detected dataset: {args.dataset}")
        
    dataset_name = args.dataset.split('.')[0]

    try:
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Data Ingestion
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  STEP 1: Data Ingestion")
        logger.info("=" * 60)
        loader = DataLoader()
        df_raw = loader.load_csv(args.dataset)
        logger.info(f"Loaded dataset: {args.dataset} | Shape: {df_raw.shape}")
        
        # Auto-detect target column
        if args.target_col is None and len(df_raw.columns) > 0:
            args.target_col = df_raw.columns[-1]
            logger.info(f"Auto-detected target column (last column): '{args.target_col}'")
            
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Data Profiling (Enhanced)
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  STEP 2: Data Profiling")
        logger.info("=" * 60)
        profiler = DataProfiler()
        profile = profiler.analyze(df_raw)
        
        # Log profiling summary
        health = profile.get("overall_health", {})
        logger.info(f"  Total cells:        {health.get('total_cells', 'N/A')}")
        logger.info(f"  Missing cells:      {health.get('total_missing_cells', 0)}")
        logger.info(f"  Placeholder cells:  {health.get('total_placeholder_cells', 0)}")
        logger.info(f"  Completeness:       {health.get('completeness_score', 0):.2%}")
        logger.info(f"  Duplicate rows:     {health.get('duplicate_rows', 0)}")
        logger.info(f"  Droppable columns:  {health.get('droppable_columns_count', 0)}")
        logger.info(f"  Splittable columns: {health.get('splittable_columns_count', 0)}")
        
        # Save Profile
        with open(f"data/processed/profile_{dataset_name}.json", "w") as f:
            json.dump(profile, f, indent=2, default=str)
        logger.info(f"Profile saved to data/processed/profile_{dataset_name}.json")
            
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Semantic Analysis (Enhanced with 5 categories)
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info(f"  STEP 3: Semantic Analysis ({args.llm_provider})")
        logger.info("=" * 60)
        llm = LLMFactory.create_client(provider=args.llm_provider)
        analyzer = SemanticAnalyzer(llm)
        
        logger.info("Analyzing profile for issues across 5 categories...")
        issues_report = analyzer.analyze_profile(profile, df=df_raw, target_col=args.target_col)
        
        if "issues" not in issues_report:
            # Fallback if LLM output structure is messy
            logger.warning("LLM report format issue, attempting to extract 'issues' list.")
            issues = issues_report if isinstance(issues_report, list) else []
        else:
            issues = issues_report["issues"]
        
        # Log category summary
        cat_summary = issues_report.get("category_summary", {})
        logger.info(f"Detected {len(issues)} issues:")
        for cat, count in cat_summary.items():
            logger.info(f"  [{cat}]: {count} issue(s)")
        
        with open(f"data/processed/issues_report_{dataset_name}.json", "w") as f:
            json.dump(issues_report, f, indent=2)
        logger.info(f"Issues report saved to data/processed/issues_report_{dataset_name}.json")

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Workflow Generation (5-phase workflow)
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  STEP 4: Generating 5-Phase Cleaning Workflow")
        logger.info("=" * 60)
        wf_gen = WorkflowGenerator(llm)
        workflow = wf_gen.generate_workflow(profile, issues, df=df_raw, target_col=args.target_col)
        
        # Log workflow phases
        steps = workflow.get("steps", [])
        phase_counts = {}
        for s in steps:
            p = s.get("phase", "other")
            phase_counts[p] = phase_counts.get(p, 0) + 1
        
        logger.info(f"Generated {len(steps)} workflow steps:")
        for phase, count in phase_counts.items():
            logger.info(f"  [{phase}]: {count} step(s)")
        
        with open(f"data/processed/cleaning_workflow_{dataset_name}.json", "w") as f:
            json.dump(workflow, f, indent=2)
        logger.info(f"Workflow saved to data/processed/cleaning_workflow_{dataset_name}.json")

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Execution (with per-category reporting)
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  STEP 5: Executing Cleaning Workflow")
        logger.info("=" * 60)
        executor = Executor()
        df_clean = executor.execute(df_raw, workflow)
        
        # Print detailed execution report
        executor.print_execution_report()
        
        # Save execution report
        exec_report = executor.get_execution_report()
        with open(f"data/processed/execution_report_{dataset_name}.json", "w") as f:
            json.dump(exec_report, f, indent=2, default=str)
        logger.info(f"Execution report saved to data/processed/execution_report_{dataset_name}.json")
        
        # Save cleaned data
        output_path = f"data/processed/clean_{args.dataset}"
        try:
            df_clean.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        except PermissionError:
            logger.error(f"Permission denied: Could not save to {output_path}. Is the file open in Excel or another program?")
            # Pipeline can continue to evaluation even if saving fails.
        logger.info(f"Shape: {df_raw.shape} -> {df_clean.shape}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Evaluation (per-category quality metrics)
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  STEP 6: Evaluation")
        logger.info("=" * 60)
        evaluator = Evaluator()
        
        quality_metrics = evaluator.evaluate_quality(df_raw, df_clean)
        ml_metrics = evaluator.evaluate_ml_performance(df_raw, df_clean, target_col=args.target_col)
        
        report = {
            "quality_metrics": quality_metrics,
            "ml_metrics": ml_metrics,
            "execution_summary": {
                phase: {
                    "executed": data["steps_executed"],
                    "skipped": data["steps_skipped"]
                }
                for phase, data in exec_report.items()
            }
        }
        
        # Log evaluation highlights
        logger.info("--- Quality Metrics ---")
        
        cat1 = quality_metrics.get("category_1_null_treatment", {})
        logger.info(f"  Nulls:      {cat1.get('total_missing_raw', 0)} -> {cat1.get('total_missing_clean', 0)} "
                     f"({cat1.get('reduction_pct', 0)}% reduction)")
        
        cat2 = quality_metrics.get("category_2_duplicate_treatment", {})
        logger.info(f"  Duplicates: {cat2.get('duplicates_raw', 0)} -> {cat2.get('duplicates_clean', 0)} "
                     f"({cat2.get('duplicates_removed', 0)} removed)")
        
        cat3 = quality_metrics.get("category_3_missing_population", {})
        logger.info(f"  Completeness: {cat3.get('completeness_raw', 0)}% -> {cat3.get('completeness_clean', 0)}%")
        
        cat4 = quality_metrics.get("category_4_column_dropping", {})
        logger.info(f"  Columns:    {cat4.get('columns_raw', 0)} -> {cat4.get('columns_clean', 0)} "
                     f"(dropped: {cat4.get('columns_dropped', [])})")
        
        cat5 = quality_metrics.get("category_5_column_splitting", {})
        logger.info(f"  New cols:   {cat5.get('new_columns_created', [])}")
        
        rows = quality_metrics.get("rows_retained", {})
        logger.info(f"  Rows:       {rows.get('raw', 0)} -> {rows.get('clean', 0)} "
                     f"({rows.get('retention_pct', 0)}% retained)")
        
        logger.info("--- ML Metrics ---")
        if ml_metrics.get("skipped"):
            logger.info(f"  ML evaluation skipped: {ml_metrics.get('reason', 'unknown')}")
        elif ml_metrics.get("error"):
            logger.info(f"  ML evaluation failed: {ml_metrics.get('error')}")
        elif ml_metrics.get("task_type") == "classification":
            logger.info(f"  Task Type:  Classification")
            logger.info(f"  Accuracy:   {ml_metrics.get('accuracy_raw', 'N/A')} -> {ml_metrics.get('accuracy_clean', 'N/A')} "
                         f"({ml_metrics.get('accuracy_improvement', 'N/A')}% improvement)")
            logger.info(f"  F1 Score:   {ml_metrics.get('f1_raw', 'N/A')} -> {ml_metrics.get('f1_clean', 'N/A')} "
                         f"({ml_metrics.get('f1_improvement', 'N/A')}% improvement)")
        elif ml_metrics.get("task_type") == "regression":
            logger.info(f"  Task Type:  Regression")
            logger.info(f"  R2 Score:   {ml_metrics.get('r2_raw', 'N/A')} -> {ml_metrics.get('r2_clean', 'N/A')} "
                         f"({ml_metrics.get('r2_improvement', 'N/A')}% improvement)")
            logger.info(f"  RMSE:       {ml_metrics.get('rmse_raw', 'N/A')} -> {ml_metrics.get('rmse_clean', 'N/A')} "
                         f"(reduced by {ml_metrics.get('rmse_improvement', 'N/A')})")
        else:
            logger.info(f"  Accuracy:   {ml_metrics.get('accuracy_raw', 'N/A')} -> {ml_metrics.get('accuracy_clean', 'N/A')} "
                         f"({ml_metrics.get('accuracy_improvement', 'N/A')}% improvement)")
            logger.info(f"  F1 Score:   {ml_metrics.get('f1_raw', 'N/A')} -> {ml_metrics.get('f1_clean', 'N/A')} "
                         f"({ml_metrics.get('f1_improvement', 'N/A')}% improvement)")
        
        with open(f"data/processed/evaluation_report_{dataset_name}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to data/processed/evaluation_report_{dataset_name}.json")

        # ===============================================================
        # STEP 7: Benchmark Accuracy (if benchmark file exists)
        # ===============================================================
        benchmark_dir = "data/benchmark"
        benchmark_path = None
        
        if os.path.isdir(benchmark_dir):
            # Try exact name match first, then any CSV in the folder
            exact_match = os.path.join(benchmark_dir, args.dataset)
            if os.path.isfile(exact_match):
                benchmark_path = exact_match
            else:
                # Also try clean_ prefix or just the dataset name
                for fname in os.listdir(benchmark_dir):
                    if fname.lower().endswith(".csv"):
                        benchmark_path = os.path.join(benchmark_dir, fname)
                        break
        
        if benchmark_path and os.path.isfile(benchmark_path):
            logger.info("=" * 60)
            logger.info("  STEP 7: Benchmark Accuracy Evaluation")
            logger.info("=" * 60)
            logger.info(f"Benchmark file found: {benchmark_path}")
            
            try:
                df_benchmark = pd.read_csv(benchmark_path)
                logger.info(f"Benchmark shape: {df_benchmark.shape}")
                
                benchmark_report = evaluator.evaluate_against_benchmark(df_clean, df_benchmark)
                
                # Log benchmark results
                logger.info("--- Benchmark Accuracy ---")
                logger.info(f"  Cell-Level Accuracy:     {benchmark_report['overall_cell_accuracy']}%")
                logger.info(f"  Row Exact Match Rate:    {benchmark_report['row_exact_match_rate']}%")
                logger.info(f"  Value Match (non-null):  {benchmark_report['value_match_accuracy']}%")
                logger.info(f"  Null Handling Accuracy:  {benchmark_report['null_handling_accuracy']}%")
                logger.info(f"  Schema F1:               {benchmark_report['schema_match']['schema_f1']}")
                
                # Log per-column accuracy
                col_acc = benchmark_report.get("column_accuracy", {})
                if col_acc:
                    logger.info("  Per-Column Accuracy:")
                    for col_name, col_data in col_acc.items():
                        acc = col_data["accuracy"]
                        mismatches = col_data["mismatches"]
                        status = "[OK]" if acc == 100.0 else f"[{mismatches} mismatches]"
                        logger.info(f"    {col_name}: {acc}% {status}")
                
                # Log schema issues if any
                schema = benchmark_report.get("schema_match", {})
                if schema.get("missing_from_cleaned"):
                    logger.warning(f"  Columns in benchmark but missing from cleaned: {schema['missing_from_cleaned']}")
                if schema.get("extra_in_cleaned"):
                    logger.info(f"  Extra columns in cleaned (not in benchmark): {schema['extra_in_cleaned']}")
                
                # Row count
                row_comp = benchmark_report.get("row_comparison", {})
                logger.info(f"  Rows compared: {row_comp.get('rows_compared', 'N/A')} "
                           f"(cleaned: {row_comp.get('cleaned_rows', 'N/A')}, "
                           f"benchmark: {row_comp.get('benchmark_rows', 'N/A')})")
                
                logger.info(f"  SUMMARY: {benchmark_report['summary']}")
                
                # Save benchmark report
                report["benchmark_accuracy"] = benchmark_report
                with open(f"data/processed/benchmark_report_{dataset_name}.json", "w") as f:
                    json.dump(benchmark_report, f, indent=2, default=str)
                logger.info(f"Benchmark report saved to data/processed/benchmark_report_{dataset_name}.json")
                
                # Also update the main evaluation report
                with open(f"data/processed/evaluation_report_{dataset_name}.json", "w") as f:
                    json.dump(report, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"Benchmark evaluation failed: {e}")
        else:
            logger.info("No benchmark file found in data/benchmark/. Skipping benchmark accuracy.")
            logger.info("  To enable: place a cleaned CSV in data/benchmark/")
            
        logger.info("=" * 60)
        logger.info("  [OK] Pipeline Completed Successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception("Pipeline failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
