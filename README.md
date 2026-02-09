# LLM-Assisted Automated Data Cleaning

## Overview
This project is an intelligent automated pipeline that detects data quality issues in tabular datasets, uses Large Language Models (LLMs) (specifically **Gemma**) for semantic analysis, and generates executable cleaning workflows.

## Features
- **Data Profiling**: Comprehensive statistical analysis of datasets.
- **Semantic Reasoning**: Uses Gemma to understand column meanings and detect semantic errors.
- **Automated Cleaning**: Generates and executes Python/Pandas code to clean data.
- **Evaluation**: Measures improvements in data quality and downstream ML model performance.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (create `.env`):
   ```
   GOOGLE_API_KEY=your_key_here
   # OR for Ollama
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## Usage
(Coming soon)
