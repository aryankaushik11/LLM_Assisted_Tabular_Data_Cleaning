# LLM-Assisted Automated Data Cleaning Pipeline

A fully autonomous, data-agnostic pipeline that detects data quality issues in raw tabular datasets, uses Large Language Models (LLMs) to perform deep semantic reasoning, generates deterministic JSON-based cleaning workflows, and rigorously evaluates its own cleaning against Ground Truth benchmarks and downstream Machine Learning models.

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-orange)

## 🧠 Core Architecture

The pipeline uses a **7-Step Autonomous Architecture**:

1. **Intelligent Ingestion (`DataLoader`)**: Auto-loads dirty datasets and automatically infers target ML columns.
2. **Contextual Profiling (`DataProfiler`)**: Scans datasets to generate complex missingness statistics, identify data outliers, and capture schema representations.
3. **Semantic Analysis (`SemanticAnalyzer`)**: LLM scans the data profile and raw statistical samples to act as an expert Data Scientist, identifying domain-specific logic flaws (e.g. invalid placeholder strings natively hiding as text, fused columns, inconsistent categoricals).
4. **Workflow Generation (`WorkflowGenerator`)**: Compiles real-world problems into a strict, programmatic 6-Phase execution plan (Treat Nulls → Drop Duplicates → Populate Missing → Drop Redundant Columns → Split Strings → Standardize Types).
5. **Robust Execution (`Executor`)**: A fail-safe pandas engine runs the execution plan. It includes advanced features like **Contextual Group Imputation** (intelligently filling nulls based on relational entity structures) and **Regex Time Processors** (to gracefully handle messy/broken Date and Time strings).
6. **Machine Learning Evaluation (`Evaluator`)**: Auto-detects Classification vs. Regression tasks, builds pipelines on both raw and cleaned data, and grades the improvement delta.
7. **Benchmark Profilomentry (`Evaluator`)**: Compares the AI-cleaned output cell-by-cell against Ground Truth benchmark files to generate final Accuracy and Schema F1 scores.

## ✨ Key Features

- ✅ **Data Agnostic**: Zero hardcoded components. Point it to any CSV and the LLM detects the feature spaces natively.
- ✅ **Contextual Missing Value Imputation**: Does not artificially average global columns. Instead, it groups relational subsets (e.g. Flight Numbers, User IDs) to impute highly precise localized values.
- ✅ **Hybrid LLM Support**: Fully supports locally-hosted models (via **Ollama**) for secure internal deployments, as well as Cloud API models (via **Google Generative AI**).
- ✅ **Regex-Powered Standardization**: Automatically sanitizes string currencies, fixes dirty am/pm parentheticals without breaking pandas datetime indices, and securely maps them into appropriate 24-hour database encodings.

## 🚀 Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/aryankaushik11/LLM_Assisted_Tabular_Data_Cleaning.git
cd LLM_Assisted_Tabular_Data_Cleaning
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure Environment:**
Create a `.env` file at the root. You can choose to use either local models (Ollama) or cloud models (Google Gemini).
```env
# For Ollama (Local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:27b

# For Google API (Cloud)
GOOGLE_API_KEY=your_key_here
```

## 💻 Usage

Place your dirty dataset into the `data/raw/` folder. Optionally, place the heavily-cleaned ground truth equivalent into `data/benchmark/` for accuracy grading.

Run the holistic pipeline with:
```bash
# Auto-detects the first dataset in data/raw
python main.py

# Or explicitly target a dataset and target ML column
python main.py --dataset flights_dirty.csv --target_col act_arr_time --llm_provider ollama
```

### Outputs
The pipeline automatically provisions detailed reporting inside `data/processed/`:
* `profile_[dataset].json` - Deep profile metrics pre-cleaning.
* `issues_report_[dataset].json` - The LLM's semantic diagnosis block.
* `cleaning_workflow_[dataset].json` - The generated 6-phase cleaning routine.
* `execution_report_[dataset].json` - The executor's play-by-play status matrix.
* `evaluation_report_[dataset].json` - ML quality improvement scores.
* `benchmark_report_[dataset].json` - Cell-level ground truth accuracy.
* `clean_[dataset].csv` - Your brand new, production-ready dataset!
