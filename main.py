import pm4py
import pandas as pd
from src.heuristics_miner import discover_heuristics_net, save_model_summary


def load_event_log(file_path):
    """
    Load an XES event log file into a pandas DataFrame.

    Parameters:
    -----------
    file_path : str
        Path to the .xes or .xes.gz event log file

    Returns:
    --------
    pd.DataFrame
        Event log as a pandas DataFrame
    """
    # Read the XES log using pm4py
    event_log = pm4py.read_xes(file_path)

    # Convert the event log to a pandas DataFrame
    df = pm4py.convert_to_dataframe(event_log)

    return df


# =============================================================================
# TEST SECTION
# =============================================================================

if __name__ == "__main__":
    # Load the Sepsis Cases event log
    file_path = "data/Sepsis Cases - Event Log.xes.gz"
    print(f"Loading event log from: {file_path}")
    print("=" * 60)

    df = load_event_log(file_path)

    # Print first few rows
    print("\n📊 First 5 rows (df.head()):")
    print(df.head())

    # Print column names
    print("\n📋 Column names (df.columns):")
    print(df.columns.tolist())

    # Print number of rows
    print(f"\n📈 Number of rows: {len(df)}")

    # Print number of unique cases
    num_unique_cases = df['case:concept:name'].nunique()
    print(f"🔢 Number of unique cases (case:concept:name): {num_unique_cases}")

    # Print number of unique activities
    num_unique_activities = df['concept:name'].nunique()
    print(f"🔢 Number of unique activities (concept:name): {num_unique_activities}")

    print("\n" + "=" * 60)
    print("✅ Event log loaded and inspected successfully!")

    print("\n" + "=" * 60)
    print("Running Heuristics Miner...")

    heuristics_result = discover_heuristics_net(
        df,
        dependency_threshold=0.5,
        frequency_threshold=2
    )

    save_model_summary(
        heuristics_result,
        "results/heuristics_net_summary.json"
    )

    print("✅ Heuristics Miner finished!")
    print("Summary saved to: results/heuristics_net_summary.json")