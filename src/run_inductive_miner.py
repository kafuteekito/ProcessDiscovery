import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inductive_miner import (
    discover_inductive_model,
    summarize_model,
    save_model_summary
)
import pm4py


def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load Sepsis Cases XES file using pm4py
    event_log = pm4py.read_xes("data/Sepsis Cases - Event Log.xes.gz")
    
    # Convert to pandas dataframe with pm4py-style column names
    df = pm4py.convert_to_dataframe(event_log)
    
    # Discover process model using inductive miner
    model = discover_inductive_model(df)
    
    # Create summary
    summary = summarize_model(model)
    
    # Save summary to JSON file
    save_model_summary(model, "results/inductive_model_summary.json")
    
    print("Inductive Miner completed successfully!")
    print(f"Summary saved to: results/inductive_model_summary.json")
    
    # Print summary to console
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()