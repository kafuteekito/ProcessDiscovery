#!/usr/bin/env python3
"""run_sepsis.py - Run Heuristic Miner on Sepsis Cases Event Log"""

import sys
import os
sys.path.insert(0, 'src')

import pm4py
from heuristics_miner import (
    extract_traces,
    get_start_activities,
    get_end_activities,
    compute_directly_follows_counts,
    compute_dependency_measure,
    filter_dependencies,
    compute_and_probability,
    detect_split_types,
    build_model_summary,
    print_model_summary,
    save_model_summary
)


def main():
    # Paths
    input_path = 'data/Sepsis Cases - Event Log.xes.gz'
    output_path = 'results/heuristics_model_summary.json'
    
    # Ensure output directory exists
    os.makedirs('results', exist_ok=True)
    
    # Load XES file using pm4py (only for file loading)
    print("=" * 70)
    print("LOADING SEPSIS EVENT LOG")
    print("=" * 70)
    print(f"\nInput file: {input_path}")
    
    event_log = pm4py.read_xes(input_path)
    df = pm4py.convert_to_dataframe(event_log)
    
    # Check required columns exist (pm4py uses different column names)
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    available_cols = list(df.columns)
    print(f"\nAvailable columns: {available_cols}")
    
    # Map common pm4py column names to our expected names if needed
    col_mapping = {
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    }
    
    for pm_col, our_col in col_mapping.items():
        if pm_col in df.columns and our_col not in df.columns:
            df[our_col] = df[pm_col]
    
    # Verify required columns now exist
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n[ERROR] Missing required columns: {missing}")
        print("Please ensure the XES file contains case ID, activity, and timestamp")
        return None
    
    print(f"\nLoaded {len(df)} events")
    print(f"Unique cases: {df['case:concept:name'].nunique()}")
    print(f"Unique activities: {df['concept:name'].nunique()}")
    
    # Run mining pipeline
    print("\n" + "=" * 70)
    print("RUNNING HEURISTIC MINER PIPELINE")
    print("=" * 70)
    
    print("\n1. Extracting traces...")
    traces = extract_traces(df)
    print(f"   Extracted {len(traces)} traces")
    
    print("\n2. Getting start/end activities...")
    starts = get_start_activities(traces)
    ends = get_end_activities(traces)
    print(f"   Start activities: {len(starts)}")
    print(f"   End activities: {len(ends)}")
    
    print("\n3. Computing directly-follows counts...")
    normal_follows, self_loops = compute_directly_follows_counts(df)
    print(f"   Normal follow relations: {len(normal_follows)}")
    print(f"   Self-loops: {len(self_loops)}")
    
    # Set thresholds
    print("\n4. Filtering dependencies...")
    thresholds = {
        'dependency_threshold': 0.5,
        'frequency_threshold': 2,
        'all_dependencies_threshold': 0.05,
        'and_threshold': 0.7,
        'xor_threshold': 0.3
    }
    
    result = filter_dependencies(
        normal_follows,
        self_loops,
        dependency_threshold=thresholds['dependency_threshold'],
        frequency_threshold=thresholds['frequency_threshold'],
        all_dependencies_threshold=thresholds['all_dependencies_threshold'],
        all_activities=True
    )
    print(f"   Kept {len(result['edges'])} edges")
    print(f"   Kept {len(result['self_loops'])} self-loops")
    print(f"   Filtered out {len(result['filtered_out'])} relations")
    
    print("\n5. Detecting split types...")
    final_split_result = detect_split_types(
        result['edges'],
        traces,
        and_threshold=thresholds['and_threshold'],
        xor_threshold=thresholds['xor_threshold']
    )
    print(f"   AND-splits: {final_split_result['summary']['AND']}")
    print(f"   XOR-splits: {final_split_result['summary']['XOR']}")
    print(f"   OR-splits: {final_split_result['summary']['OR']}")
    
    # Build and save model summary
    print("\n" + "=" * 70)
    print("GENERATING MODEL SUMMARY")
    print("=" * 70)
    
    summary = build_model_summary(
        edges=result['edges'],
        self_loops=result['self_loops'],
        start_activities=starts,
        end_activities=ends,
        split_result=final_split_result,
        thresholds=thresholds,
        all_activities=result['all_activities']
    )
    
    print_model_summary(summary)
    save_model_summary(summary, output_path)
    print(f"\nModel saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    main()