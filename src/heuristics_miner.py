"""
heuristics_miner.py

Heuristics Miner implementation for Process Discovery.
This module provides functions to discover a process model from an event log
using the Heuristics Miner algorithm, which is based on directly-follows graphs
and dependency measures.

Author: Process Mining Course Project
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def extract_traces(df: pd.DataFrame) -> List[List[str]]:
    """
    Extract traces from an event log DataFrame.
    
    This function sorts events by case ID and timestamp, then groups them
    by case ID to create traces (sequences of activities).
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log DataFrame with at least these columns:
        - case:concept:name : Case identifier
        - concept:name : Activity name
        - time:timestamp : Event timestamp
    
    Returns
    -------
    List[List[str]]
        List of traces, where each trace is a list of activity names
        in chronological order.
    
    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'case:concept:name': ['A', 'A', 'A', 'B', 'B'],
    ...     'concept:name': ['Start', 'Process', 'End', 'Start', 'End'],
    ...     'time:timestamp': pd.to_datetime([
    ...         '2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:10',
    ...         '2024-01-01 10:00', '2024-01-01 10:05'
    ...     ])
    ... })
    >>> traces = extract_traces(df)
    >>> traces
    [['Start', 'Process', 'End'], ['Start', 'End']]
    """
    # Validate required columns
    required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing_columns}. "
            f"Required columns are: {required_columns}"
        )
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Sort by case ID and timestamp
    df_copy = df_copy.sort_values(
        by=['case:concept:name', 'time:timestamp']
    ).reset_index(drop=True)
    
    # Group by case ID and collect activity names
    traces = []
    for case_id, group in df_copy.groupby('case:concept:name', sort=False):
        trace = group['concept:name'].tolist()
        traces.append(trace)
    
    return traces


def compute_activity_frequencies(traces: List[List[str]]) -> Dict[str, int]:
    """
    Compute the frequency of each activity across all traces.
    
    Parameters
    ----------
    traces : List[List[str]]
        List of traces, where each trace is a list of activity names.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping activity names to their occurrence counts.
    
    Examples
    --------
    >>> traces = [['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'B', 'C']]
    >>> freq = compute_activity_frequencies(traces)
    >>> freq['A']
    3
    >>> freq['B']
    3
    >>> freq['C']
    2
    """
    frequencies = defaultdict(int)
    
    for trace in traces:
        for activity in trace:
            frequencies[activity] += 1
    
    return dict(frequencies)


def compute_start_end_activities(traces: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compute the frequency of start and end activities across all traces.
    
    A start activity is the first activity in a trace, and an end activity
    is the last activity in a trace.
    
    Parameters
    ----------
    traces : List[List[str]]
        List of traces, where each trace is a list of activity names.
    
    Returns
    -------
    Tuple[Dict[str, int], Dict[str, int]]
        A tuple containing:
        - start_activities : Dictionary mapping start activity names to their counts.
        - end_activities : Dictionary mapping end activity names to their counts.
    
    Examples
    --------
    >>> traces = [['A', 'B', 'C'], ['A', 'D'], ['B', 'C']]
    >>> starts, ends = compute_start_end_activities(traces)
    >>> starts['A']
    2
    >>> starts['B']
    1
    >>> ends['C']
    2
    >>> ends['D']
    1
    """
    start_activities = defaultdict(int)
    end_activities = defaultdict(int)
    
    for trace in traces:
        if len(trace) > 0:
            # First activity is the start activity
            start_activities[trace[0]] += 1
            # Last activity is the end activity
            end_activities[trace[-1]] += 1
    
    return dict(start_activities), dict(end_activities)


def compute_directly_follows(traces: List[List[str]]) -> Dict[Tuple[str, str], int]:
    """
    Compute directly-follows relations between activities.
    
    A directly-follows relation (A > B) exists when activity A is immediately
    followed by activity B in at least one trace.
    
    Parameters
    ----------
    traces : List[List[str]]
        List of traces, where each trace is a list of activity names.
    
    Returns
    -------
    Dict[Tuple[str, str], int]
        Dictionary mapping (source_activity, target_activity) tuples to their
        occurrence counts.
    
    Examples
    --------
    >>> traces = [['A', 'B', 'C'], ['A', 'B', 'D'], ['B', 'C', 'D']]
    >>> dfg = compute_directly_follows(traces)
    >>> dfg[('A', 'B')]
    2
    >>> dfg[('B', 'C')]
    2
    >>> dfg[('C', 'D')]
    1
    """
    directly_follows = defaultdict(int)
    
    for trace in traces:
        # Iterate through consecutive pairs
        for i in range(len(trace) - 1):
            source = trace[i]
            target = trace[i + 1]
            directly_follows[(source, target)] += 1
    
    return dict(directly_follows)


def compute_dependency_measures(
    dfg_counts: Dict[Tuple[str, str], int]
) -> Dict[Tuple[str, str], float]:
    """
    Compute dependency measures for all activity pairs.
    
    The dependency measure indicates the strength and direction of the
    relationship between two activities. It ranges from -1 to 1.
    
    The formula is:
        dependency(A, B) = (freq(A>B) - freq(B>A)) / (freq(A>B) + freq(B>A) + 1)
    
    A positive dependency close to 1 indicates that A is likely followed by B,
    while a negative dependency close to -1 indicates the opposite.
    
    Parameters
    ----------
    dfg_counts : Dict[Tuple[str, str], int]
        Dictionary mapping (source_activity, target_activity) tuples to their
        occurrence counts (directly-follows graph).
    
    Returns
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping (source_activity, target_activity) tuples to their
        dependency measures.
    
    Examples
    --------
    >>> dfg_counts = {
    ...     ('A', 'B'): 100,
    ...     ('B', 'A'): 10,
    ...     ('B', 'C'): 50,
    ...     ('C', 'B'): 5
    ... }
    >>> deps = compute_dependency_measures(dfg_counts)
    >>> round(deps[('A', 'B')], 2)
    0.82
    >>> round(deps[('B', 'C')], 2)
    0.82
    """
    dependency_measures = {}
    
    # Get all unique pairs of activities
    all_pairs = set()
    for (source, target) in dfg_counts.keys():
        all_pairs.add((source, target))
        all_pairs.add((target, source))
    
    for source, target in all_pairs:
        freq_forward = dfg_counts.get((source, target), 0)
        freq_backward = dfg_counts.get((target, source), 0)
        
        # Compute dependency measure using the formula
        denominator = freq_forward + freq_backward + 1
        dependency = (freq_forward - freq_backward) / denominator
        
        dependency_measures[(source, target)] = dependency
    
    return dependency_measures


def discover_heuristics_net(
    df: pd.DataFrame,
    dependency_threshold: float = 0.5,
    frequency_threshold: int = 2
) -> Dict[str, Any]:
    """
    Discover a Heuristics Net process model from an event log.
    
    This function applies the Heuristics Miner algorithm to discover a process
    model. It uses directly-follows relations and dependency measures to identify
    significant relationships between activities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log DataFrame with at least these columns:
        - case:concept:name : Case identifier
        - concept:name : Activity name
        - time:timestamp : Event timestamp
    
    dependency_threshold : float, optional
        Minimum dependency measure for an edge to be included (default: 0.5).
        The dependency measure ranges from -1 to 1, so values > 0 indicate
        that the forward relation is stronger than the backward relation.
    
    frequency_threshold : int, optional
        Minimum directly-follows frequency for an edge to be included
        (default: 2).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the discovered process model with keys:
        - activities : Set of unique activity names.
        - activity_frequencies : Dictionary of activity occurrence counts.
        - start_activities : Dictionary of start activity counts.
        - end_activities : Dictionary of end activity counts.
        - directly_follows_counts : Dictionary of directly-follows counts.
        - dependency_measures : Dictionary of dependency measures.
        - selected_edges : List of edges that passed the thresholds.
        - thresholds_used : Dictionary of thresholds applied.
        - model_summary : Summary dictionary of the model.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'case:concept:name': ['A', 'A', 'A', 'B', 'B'],
    ...     'concept:name': ['Start', 'Process', 'End', 'Start', 'End'],
    ...     'time:timestamp': pd.to_datetime([
    ...         '2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:10',
    ...         '2024-01-01 10:00', '2024-01-01 10:05'
    ...     ])
    ... })
    >>> model = discover_heuristics_net(df)
    >>> model['activities']
    {'Start', 'Process', 'End'}
    """
    # Extract traces from the event log
    traces = extract_traces(df)
    
    # Compute activity frequencies
    activity_frequencies = compute_activity_frequencies(traces)
    
    # Compute start and end activities
    start_activities, end_activities = compute_start_end_activities(traces)
    
    # Compute directly-follows relations
    directly_follows_counts = compute_directly_follows(traces)
    
    # Compute dependency measures
    dependency_measures = compute_dependency_measures(directly_follows_counts)
    
    # Get all unique activities
    activities = set(activity_frequencies.keys())
    
    # Select edges based on thresholds
    selected_edges = []
    for (source, target), freq in directly_follows_counts.items():
        dependency = dependency_measures.get((source, target), 0)
        
        # Keep only edges that pass both thresholds
        if freq >= frequency_threshold and dependency >= dependency_threshold:
            selected_edges.append({
                'source': source,
                'target': target,
                'frequency': freq,
                'dependency': dependency
            })
    
    # Create the model summary
    model_summary = summarize_model({
        'activities': activities,
        'activity_frequencies': activity_frequencies,
        'start_activities': start_activities,
        'end_activities': end_activities,
        'directly_follows_counts': directly_follows_counts,
        'dependency_measures': dependency_measures,
        'selected_edges': selected_edges,
        'thresholds_used': {
            'dependency_threshold': dependency_threshold,
            'frequency_threshold': frequency_threshold
        }
    })
    
    # Return the complete model
    return {
        'activities': activities,
        'activity_frequencies': activity_frequencies,
        'start_activities': start_activities,
        'end_activities': end_activities,
        'directly_follows_counts': directly_follows_counts,
        'dependency_measures': dependency_measures,
        'selected_edges': selected_edges,
        'thresholds_used': {
            'dependency_threshold': dependency_threshold,
            'frequency_threshold': frequency_threshold
        },
        'model_summary': model_summary
    }


def summarize_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of the discovered process model.
    
    Parameters
    ----------
    model : Dict[str, Any]
        The discovered process model from discover_heuristics_net.
    
    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing:
        - number_of_activities : Total number of unique activities.
        - number_of_edges : Total number of selected edges.
        - number_of_start_activities : Number of unique start activities.
        - number_of_end_activities : Number of unique end activities.
        - dependency_threshold : Dependency threshold used.
        - frequency_threshold : Frequency threshold used.
    
    Examples
    --------
    >>> model = {
    ...     'activities': {'A', 'B', 'C'},
    ...     'activity_frequencies': {'A': 3, 'B': 3, 'C': 2},
    ...     'start_activities': {'A': 3},
    ...     'end_activities': {'C': 2},
    ...     'selected_edges': [
    ...         {'source': 'A', 'target': 'B', 'frequency': 3, 'dependency': 0.8}
    ...     ],
    ...     'thresholds_used': {'dependency_threshold': 0.5, 'frequency_threshold': 2}
    ... }
    >>> summary = summarize_model(model)
    >>> summary['number_of_activities']
    3
    >>> summary['number_of_edges']
    1
    """
    # Extract basic statistics
    number_of_activities = len(model.get('activities', set()))
    number_of_edges = len(model.get('selected_edges', []))
    
    # Count unique start and end activities
    start_activities = model.get('start_activities', {})
    end_activities = model.get('end_activities', {})
    
    number_of_start_activities = len(start_activities)
    number_of_end_activities = len(end_activities)
    
    # Get thresholds used
    thresholds_used = model.get('thresholds_used', {})
    dependency_threshold = thresholds_used.get('dependency_threshold', 0.0)
    frequency_threshold = thresholds_used.get('frequency_threshold', 0)
    
    return {
        'number_of_activities': number_of_activities,
        'number_of_edges': number_of_edges,
        'number_of_start_activities': number_of_start_activities,
        'number_of_end_activities': number_of_end_activities,
        'dependency_threshold': dependency_threshold,
        'frequency_threshold': frequency_threshold
    }


def save_model_summary(model: Dict[str, Any], output_path: str) -> None:
    """
    Save the model summary and selected edges to a JSON file.
    
    Parameters
    ----------
    model : Dict[str, Any]
        The discovered process model from discover_heuristics_net.
    
    output_path : str
        Path to the output JSON file.
    
    Raises
    ------
    IOError
        If the file cannot be written.
    
    Examples
    --------
    >>> model = discover_heuristics_net(df)
    >>> save_model_summary(model, 'model_summary.json')
    """
    try:
        # Create the output dictionary
        output = {
            'model_summary': model.get('model_summary', summarize_model(model)),
            'selected_edges': model.get('selected_edges', []),
            'activities': list(model.get('activities', set())),
            'activity_frequencies': model.get('activity_frequencies', {}),
            'start_activities': model.get('start_activities', {}),
            'end_activities': model.get('end_activities', {}),
            'thresholds_used': model.get('thresholds_used', {})
        }
        
        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Model summary saved to: {output_path}")
    
    except IOError as e:
        print(f"Error saving model summary: {e}")
        raise


def load_xes_as_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load an XES event log file and convert it to a pandas DataFrame.
    
    This is a helper function that uses pm4py to load the event log.
    
    Parameters
    ----------
    file_path : str
        Path to the XES file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with event data.
    """
    try:
        import pm4py
        
        # Load the XES file using pm4py
        df = pm4py.read_xes(file_path)
        
        # Convert to DataFrame if needed
        if isinstance(df, pd.DataFrame):
            return df
        else:
            # If it's an event log object, convert to DataFrame
            return pm4py.convert_to_dataframe(df)
    
    except ImportError:
        print("pm4py not installed. Please install it using: pip install pm4py")
        raise


# =============================================================================
# Test Log and Main Execution
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block with a small artificial test log.
    
    This demonstrates the functionality of the Heuristics Miner implementation.
    """
    
    print("=" * 70)
    print("HEURISTICS MINER - TEST EXECUTION")
    print("=" * 70)
    
    # Create a small artificial test log
    # This represents a simple process with:
    # - Start -> A -> B -> C -> End (common path)
    # - Start -> A -> B -> D -> End (alternative path)
    # - Start -> A -> B -> D -> E -> End (longer path)
    
    test_data = {
        'case:concept:name': [
            # Case 1: Complete trace
            'Case_001', 'Case_001', 'Case_001', 'Case_001', 'Case_001',
            # Case 2: Alternative trace
            'Case_002', 'Case_002', 'Case_002', 'Case_002',
            # Case 3: Shorter trace
            'Case_003', 'Case_003', 'Case_003',
            # Case 4: Another complete trace
            'Case_004', 'Case_004', 'Case_004', 'Case_004', 'Case_004',
            # Case 5: Repeat case
            'Case_005', 'Case_005', 'Case_005', 'Case_005', 'Case_005',
        ],
        'concept:name': [
            # Case 1
            'Start', 'Register', 'Examine', 'Treat', 'Release',
            # Case 2
            'Start', 'Register', 'Examine', 'Release',
            # Case 3
            'Register', 'Examine', 'Release',
            # Case 4
            'Start', 'Register', 'Examine', 'Treat', 'Release',
            # Case 5
            'Start', 'Register', 'Examine', 'Treat', 'Release',
        ],
        'time:timestamp': pd.to_datetime([
            # Case 1
            '2024-01-01 09:00', '2024-01-01 09:05', '2024-01-01 09:15',
            '2024-01-01 09:30', '2024-01-01 10:00',
            # Case 2
            '2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:15',
            '2024-01-01 10:45',
            # Case 3
            '2024-01-01 10:30', '2024-01-01 10:35', '2024-01-01 11:00',
            # Case 4
            '2024-01-01 11:00', '2024-01-01 11:05', '2024-01-01 11:15',
            '2024-01-01 11:30', '2024-01-01 12:00',
            # Case 5
            '2024-01-01 12:00', '2024-01-01 12:05', '2024-01-01 12:15',
            '2024-01-01 12:30', '2024-01-01 13:00',
        ])
    }
    
    # Create DataFrame
    test_df = pd.DataFrame(test_data)
    
    print("\n1. Testing extract_traces()")
    print("-" * 40)
    traces = extract_traces(test_df)
    print(f"Number of traces: {len(traces)}")
    for i, trace in enumerate(traces):
        print(f"  Trace {i+1}: {' -> '.join(trace)}")
    
    print("\n2. Testing compute_activity_frequencies()")
    print("-" * 40)
    activity_freq = compute_activity_frequencies(traces)
    for activity, freq in sorted(activity_freq.items(), key=lambda x: -x[1]):
        print(f"  {activity}: {freq} occurrences")
    
    print("\n3. Testing compute_start_end_activities()")
    print("-" * 40)
    starts, ends = compute_start_end_activities(traces)
    print("Start activities:")
    for activity, freq in sorted(starts.items(), key=lambda x: -x[1]):
        print(f"  {activity}: {freq} times")
    print("End activities:")
    for activity, freq in sorted(ends.items(), key=lambda x: -x[1]):
        print(f"  {activity}: {freq} times")
    
    print("\n4. Testing compute_directly_follows()")
    print("-" * 40)
    dfg_counts = compute_directly_follows(traces)
    print("Directly-follows relations:")
    for (src, tgt), freq in sorted(dfg_counts.items(), key=lambda x: -x[1]):
        print(f"  {src} -> {tgt}: {freq} times")
    
    print("\n5. Testing compute_dependency_measures()")
    print("-" * 40)
    dependencies = compute_dependency_measures(dfg_counts)
    print("Dependency measures (selected pairs):")
    selected_pairs = [('Start', 'Register'), ('Register', 'Examine'),
                      ('Examine', 'Treat'), ('Treat', 'Release')]
    for pair in selected_pairs:
        if pair in dependencies:
            print(f"  {pair[0]} -> {pair[1]}: {dependencies[pair]:.3f}")
    
    print("\n6. Testing discover_heuristics_net()")
    print("-" * 40)
    model = discover_heuristics_net(
        test_df,
        dependency_threshold=0.5,
        frequency_threshold=2
    )
    
    print("\nDiscovered Model Summary:")
    print(f"  Number of activities: {model['model_summary']['number_of_activities']}")
    print(f"  Number of edges: {model['model_summary']['number_of_edges']}")
    print(f"  Number of start activities: {model['model_summary']['number_of_start_activities']}")
    print(f"  Number of end activities: {model['model_summary']['number_of_end_activities']}")
    
    print("\nSelected edges (dependency >= 0.5, frequency >= 2):")
    for edge in model['selected_edges']:
        print(f"  {edge['source']} -> {edge['target']} "
              f"(freq: {edge['frequency']}, dep: {edge['dependency']:.2f})")
    
    print("\n7. Testing save_model_summary()")
    print("-" * 40)
    # Save to a temporary JSON file
    save_model_summary(model, 'test_model_summary.json')
    
    # Clean up the test file
    import os
    if os.path.exists('test_model_summary.json'):
        os.remove('test_model_summary.json')
        print("Test file cleaned up.")
    
    print("\n" + "=" * 70)
    print("TEST EXECUTION COMPLETE")
    print("=" * 70)