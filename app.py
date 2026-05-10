import os
import sys
import json
import time
import pandas as pd
import streamlit as st
import subprocess
import urllib.request
from urllib.error import URLError
from io import StringIO
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from ingestion import DataLoader
from profiling import DataProfiler
from llm_client import LLMFactory
from semantic_analyzer import SemanticAnalyzer
from workflow_generator import WorkflowGenerator
from executor import Executor
from evaluation import Evaluator

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/benchmark", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="LLM Data Cleaner",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def start_ollama():
    """Starts the ollama serve command in the background if not already running."""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        print("Ollama is already running.")
    except (URLError, ConnectionResetError, TimeoutError):
        print("Starting Ollama server...")
        try:
            # Start process in background
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            # Give it a few seconds to initialize
            time.sleep(3)
        except Exception as e:
            print(f"Failed to start Ollama: {e}")

# Call the function once when the app starts
start_ollama()

# Modern UI CSS
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --background: #0f172a;
        --card-bg: #1e293b;
        --text: #f8fafc;
        --text-muted: #94a3b8;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text);
    }
    
    .css-1d391kg {
        background-color: var(--card-bg);
    }
    
    h1, h2, h3 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .step-card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ AI-Powered Data Cleaning Pipeline")
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Automate data wrangling, profiling, and cleaning with LLMs</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    llm_provider = st.selectbox("LLM Provider", ["ollama", "google"], index=0)
    target_col = st.text_input("Target Column (Optional)", help="Leave blank to auto-detect the last column")
    if not target_col:
        target_col = None
        
    st.markdown("---")
    st.info("💡 Upload a dirty dataset to begin. Optionally upload a benchmark to evaluate accuracy.")

# Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Dirty Dataset")
    dirty_file = st.file_uploader("Choose a CSV file", type="csv", key="dirty")

with col2:
    st.subheader("📁 Upload Benchmark Dataset (Optional)")
    benchmark_file = st.file_uploader("Choose a CSV file", type="csv", key="benchmark")

if dirty_file:
    # Generate unique filenames to avoid PermissionError if file is open elsewhere
    timestamp = int(time.time())
    unique_dirty_name = f"{timestamp}_{dirty_file.name}"
    
    # Save files to appropriate directories
    dirty_path = os.path.join("data", "raw", unique_dirty_name)
    with open(dirty_path, "wb") as f:
        f.write(dirty_file.getbuffer())
        
    benchmark_path = None
    if benchmark_file:
        unique_benchmark_name = f"benchmark_{timestamp}_{benchmark_file.name}"
        benchmark_path = os.path.join("data", "benchmark", unique_benchmark_name)
        with open(benchmark_path, "wb") as f:
            f.write(benchmark_file.getbuffer())
            
    if st.button("🚀 Start Cleaning Pipeline", use_container_width=True):
        dataset_name = unique_dirty_name.split('.')[0]
        
        # UI Elements for progress
        progress_bar = st.progress(0)
        status_container = st.container()
        
        # Function to render step
        def update_step(step_num, title, description):
            with status_container:
                st.markdown(f"""
                <div class="step-card">
                    <h3 style="margin-top: 0;">Step {step_num}: {title}</h3>
                    <p style="color: #94a3b8; margin-bottom: 0;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
            progress_bar.progress(step_num / 7.0)

        try:
            # STEP 1: Data Ingestion
            update_step(1, "Data Ingestion", f"Loading dataset '{dirty_file.name}'...")
            loader = DataLoader()
            df_raw = loader.load_csv(unique_dirty_name)
            
            if target_col is None and len(df_raw.columns) > 0:
                current_target_col = df_raw.columns[-1]
            else:
                current_target_col = target_col
                
            time.sleep(1) # Visual effect
            
            # STEP 2: Data Profiling
            update_step(2, "Data Profiling", "Analyzing column distributions, missing values, and anomalies...")
            profiler = DataProfiler()
            profile = profiler.analyze(df_raw)
            with open(f"data/processed/profile_{dataset_name}.json", "w") as f:
                json.dump(profile, f, indent=2, default=str)
            time.sleep(1)
            
            # STEP 3: Semantic Analysis
            update_step(3, "Semantic Analysis", f"Using {llm_provider.upper()} to detect complex issues...")
            llm = LLMFactory.create_client(provider=llm_provider)
            analyzer = SemanticAnalyzer(llm)
            issues_report = analyzer.analyze_profile(profile, df=df_raw, target_col=current_target_col)
            
            if "issues" not in issues_report:
                issues = issues_report if isinstance(issues_report, list) else []
            else:
                issues = issues_report["issues"]
            with open(f"data/processed/issues_report_{dataset_name}.json", "w") as f:
                json.dump(issues_report, f, indent=2)
                
            # STEP 4: Workflow Generation
            update_step(4, "Generating 5-Phase Cleaning Workflow", "Designing the optimal sequence of cleaning operations...")
            wf_gen = WorkflowGenerator(llm)
            workflow = wf_gen.generate_workflow(profile, issues, df=df_raw, target_col=current_target_col)
            with open(f"data/processed/cleaning_workflow_{dataset_name}.json", "w") as f:
                json.dump(workflow, f, indent=2)
                
            # STEP 5: Executing Cleaning Workflow
            update_step(5, "Executing Cleaning Workflow", "Applying transformations, imputations, and structural fixes...")
            executor = Executor()
            df_clean = executor.execute(df_raw, workflow)
            exec_report = executor.get_execution_report()
            with open(f"data/processed/execution_report_{dataset_name}.json", "w") as f:
                json.dump(exec_report, f, indent=2, default=str)
                
            output_path = f"data/processed/clean_{unique_dirty_name}"
            try:
                df_clean.to_csv(output_path, index=False)
            except PermissionError:
                # Fallback if somehow still locked
                output_path = f"data/processed/clean_{int(time.time())}_{dirty_file.name}"
                df_clean.to_csv(output_path, index=False)
            
            # STEP 6: Evaluation
            update_step(6, "Evaluation", "Calculating quality metrics and machine learning impact...")
            evaluator = Evaluator()
            quality_metrics = evaluator.evaluate_quality(df_raw, df_clean)
            ml_metrics = evaluator.evaluate_ml_performance(df_raw, df_clean, target_col=current_target_col)
            
            report = {
                "quality_metrics": quality_metrics,
                "ml_metrics": ml_metrics,
                "execution_summary": {
                    phase: {"executed": data["steps_executed"], "skipped": data["steps_skipped"]}
                    for phase, data in exec_report.items()
                }
            }
            with open(f"data/processed/evaluation_report_{dataset_name}.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
                
            # STEP 7: Benchmark Accuracy
            if benchmark_file:
                update_step(7, "Benchmark Accuracy Evaluation", "Comparing results against ground truth benchmark...")
                df_benchmark = pd.read_csv(benchmark_path)
                benchmark_report = evaluator.evaluate_against_benchmark(df_clean, df_benchmark)
                report["benchmark_accuracy"] = benchmark_report
                with open(f"data/processed/benchmark_report_{dataset_name}.json", "w") as f:
                    json.dump(benchmark_report, f, indent=2, default=str)
                with open(f"data/processed/evaluation_report_{dataset_name}.json", "w") as f:
                    json.dump(report, f, indent=2, default=str)
            else:
                progress_bar.progress(1.0) # Finish progress

            st.success("🎉 Pipeline Completed Successfully!")
            st.balloons()
            
            # -----------------------------------------------------------
            # DISPLAY RESULTS
            # -----------------------------------------------------------
            st.markdown("---")
            st.header("📊 Results & Comparison")
            
            tab1, tab2, tab3 = st.tabs(["Data Comparison", "Quality Report", "ML Impact"])
            
            with tab1:
                st.markdown("### First 25 Columns View")
                cols_to_show = 25
                
                st.markdown("#### Dirty Dataset")
                st.dataframe(df_raw.iloc[:, :cols_to_show].head(100), use_container_width=True)
                
                st.markdown("#### Cleaned Dataset")
                st.dataframe(df_clean.iloc[:, :cols_to_show].head(100), use_container_width=True)
                
                if benchmark_file:
                    st.markdown("#### Benchmark Dataset")
                    st.dataframe(df_benchmark.iloc[:, :cols_to_show].head(100), use_container_width=True)
                    
            with tab2:
                st.markdown("### Quality Improvements")
                
                cat1 = quality_metrics.get("category_1_null_treatment", {})
                cat2 = quality_metrics.get("category_2_duplicate_treatment", {})
                cat3 = quality_metrics.get("category_3_missing_population", {})
                rows = quality_metrics.get("rows_retained", {})
                
                mc1, mc2, mc3, mc4 = st.columns(4)
                
                with mc1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: var(--text-muted);">Null Reduction</div>
                        <div class="metric-value">{cat1.get('reduction_pct', 0)}%</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">{cat1.get('total_missing_raw', 0)} → {cat1.get('total_missing_clean', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with mc2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: var(--text-muted);">Completeness</div>
                        <div class="metric-value">{cat3.get('completeness_clean', 0)}%</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Was {cat3.get('completeness_raw', 0)}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with mc3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: var(--text-muted);">Duplicates Removed</div>
                        <div class="metric-value">{cat2.get('duplicates_removed', 0)}</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Remaining: {cat2.get('duplicates_clean', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with mc4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: var(--text-muted);">Row Retention</div>
                        <div class="metric-value">{rows.get('retention_pct', 0)}%</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">{rows.get('raw', 0)} → {rows.get('clean', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                if benchmark_file:
                    st.markdown("### Benchmark Accuracy")
                    br = benchmark_report
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Cell-Level Accuracy", f"{br.get('overall_cell_accuracy', 0)}%")
                    bc2.metric("Row Exact Match", f"{br.get('row_exact_match_rate', 0)}%")
                    bc3.metric("Null Handling", f"{br.get('null_handling_accuracy', 0)}%")
            
            with tab3:
                st.markdown("### Downstream ML Performance")
                
                if ml_metrics.get("skipped"):
                    st.warning(f"ML evaluation skipped: {ml_metrics.get('reason', 'unknown')}")
                elif ml_metrics.get("error"):
                    st.error(f"ML evaluation failed: {ml_metrics.get('error')}")
                else:
                    task_type = ml_metrics.get("task_type", "classification")
                    st.info(f"Task Type Detected: **{task_type.capitalize()}**")
                    
                    mc_ml1, mc_ml2 = st.columns(2)
                    
                    if task_type == "classification":
                        mc_ml1.metric(
                            "Model Accuracy", 
                            f"{ml_metrics.get('accuracy_clean', 0)}", 
                            f"{ml_metrics.get('accuracy_improvement', 0)}% vs Raw"
                        )
                        mc_ml2.metric(
                            "F1 Score", 
                            f"{ml_metrics.get('f1_clean', 0)}", 
                            f"{ml_metrics.get('f1_improvement', 0)}% vs Raw"
                        )
                    else:
                        mc_ml1.metric(
                            "R2 Score", 
                            f"{ml_metrics.get('r2_clean', 0)}", 
                            f"{ml_metrics.get('r2_improvement', 0)}% vs Raw"
                        )
                        mc_ml2.metric(
                            "RMSE", 
                            f"{ml_metrics.get('rmse_clean', 0)}", 
                            f"Reduced by {ml_metrics.get('rmse_improvement', 0)} vs Raw",
                            delta_color="inverse"
                        )

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
