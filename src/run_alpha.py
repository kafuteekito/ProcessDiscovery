"""
Alpha Miner Algorithm - Single Entry Point

This script discovers a Petri net from an event log using the Alpha Miner algorithm
and saves the results (summary JSON and visualization PNG) to the results/ folder.

Usage:
    python run_alpha.py
"""

import os
from alpha_miner import discover_alpha_net, save_alpha_summary, visualize_alpha_net


# Configuration
EVENT_LOG_PATH = "data/Sepsis Cases - Event Log.xes.gz"  # Path to your event log file
OUTPUT_DIR = "results"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "alpha_model_summary.json")
VISUALIZATION_FILE = os.path.join(OUTPUT_DIR, "alpha_petri_net.png")


def load_event_log(file_path):
    """
    Load an event log from XES file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the XES event log file (supports .xes and .xes.gz)
    
    Returns:
    --------
    pd.DataFrame
        Event log DataFrame with columns: 'case:concept:name', 'concept:name', 'time:timestamp'
    """
    import pm4py
    df = pm4py.read_xes(file_path)
    return df


def main():
    """Main entry point for the Alpha Miner algorithm."""
    print("=" * 60)
    print("Alpha Miner Algorithm - Petri Net Discovery")
    print("=" * 60)
    
    # Step 1: Load event log
    print(f"\n[1/4] Loading event log from '{EVENT_LOG_PATH}'...")
    df = load_event_log(EVENT_LOG_PATH)
    print(f"      Loaded {len(df)} events")
    print(f"      Unique cases: {df['case:concept:name'].nunique()}")
    print(f"      Unique activities: {df['concept:name'].nunique()}")
    
    # Step 2: Discover Petri net using Alpha Miner
    print(f"\n[2/4] Discovering Petri net using Alpha Miner algorithm...")
    net, initial_marking, final_marking = discover_alpha_net(df)
    print(f"      Places: {len(net.places)}")
    print(f"      Transitions: {len(net.transitions)}")
    print(f"      Arcs: {len(net.arcs)}")
    
    # Step 3: Save model summary
    print(f"\n[3/4] Saving model summary to '{SUMMARY_FILE}'...")
    save_alpha_summary(net, initial_marking, final_marking, SUMMARY_FILE)
    print(f"      Summary saved successfully")
    
    # Step 4: Save visualization
    print(f"\n[4/4] Saving Petri net visualization to '{VISUALIZATION_FILE}'...")
    visualize_alpha_net(net, initial_marking, final_marking, VISUALIZATION_FILE)
    print(f"      Visualization saved successfully")
    
    # Completion message
    print("\n" + "=" * 60)
    print("Alpha Miner completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Model summary: {SUMMARY_FILE}")
    print(f"  - Visualization: {VISUALIZATION_FILE}")
    print()


if __name__ == "__main__":
    main()