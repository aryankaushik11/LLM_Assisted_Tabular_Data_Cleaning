import sys
import os
import json

# Add src to pythonpath
sys.path.append(os.path.abspath("src"))

from ingestion import DataLoader
from profiling import DataProfiler

def main():
    print("Testing Data Ingestion...")
    loader = DataLoader(data_path="data/raw")
    try:
        df = loader.load_csv("adult.csv")
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        
        print("\nTesting Data Profiling...")
        profiler = DataProfiler()
        profile = profiler.analyze(df)
        
        # Save profile to inspect
        with open("data/processed/profile_adult.json", "w") as f:
            f.write(json.dumps(profile, indent=2, default=str))
            
        print("Profile generated and saved to 'data/processed/profile_adult.json'")
        print("Sample Column Analysis (Age):")
        print(json.dumps(profile["column_analysis"].get("age", {}), indent=2))
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
