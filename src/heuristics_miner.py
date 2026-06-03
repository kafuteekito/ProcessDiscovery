import pandas as pd
import json

def extract_traces(df):
    """
    Extract traces from an event log DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Event log with columns: 'case:concept:name', 'concept:name', 'time:timestamp'
    
    Returns:
    --------
    list of list
        Each element is a list of activity names (str) for one trace
    """
    # Group by case (without sorting the groups themselves)
    traces = []
    
    for case_id, group in df.groupby('case:concept:name', sort=False):
        # Sort within each case by timestamp (stable sort preserves original 
        # row order for ties), then extract activity names
        trace = group.sort_values('time:timestamp', kind='stable')['concept:name'].tolist()
        traces.append(trace)
    
    return traces

def get_activity_set(traces):
    """
    Collect all unique activity names from all traces.
    
    Parameters:
    -----------
    traces : list of list (or tuple)
        List where each element is a trace (list/tuple of activity names)
    
    Returns:
    --------
    set of str
        All unique activity names found across the traces
    """
    activity_set = set()
    
    for trace in traces:
        for activity in trace:
            activity_set.add(activity)
    
    return activity_set

def get_start_activities(traces):
    """Get all unique starting activities from traces."""
    return {trace[0] for trace in traces}


def get_end_activities(traces):
    """Get all unique ending activities from traces."""
    return {trace[-1] for trace in traces}

def compute_directly_follows(traces):
    """
    Compute directly-follows relationships between consecutive activities.
    
    Parameters:
    -----------
    traces : list of list (or tuple)
        List where each element is a trace (list/tuple of activity names)
    
    Returns:
    --------
    dict
        Keys are (from_activity, to_activity) tuples
        Values are the count of how often each pair appears
    """
    directly_follows = {}
    
    for trace in traces:
        for i in range(len(trace) - 1):
            pair = (trace[i], trace[i + 1])
            directly_follows[pair] = directly_follows.get(pair, 0) + 1
    
    return directly_follows

def compute_dependency_measures(dfg_counts):
    """Compute dependency measures for directly-follows pairs."""
    return {
        (act_from, act_to): (freq - dfg_counts.get((act_to, act_from), 0)) 
                            / (freq + dfg_counts.get((act_to, act_from), 0) + 1)
        for (act_from, act_to), freq in dfg_counts.items()
    }

def select_edges(dfg_counts, dependency_measures, dependency_threshold=0.5, frequency_threshold=1):
    
    selected = []
    
    for (act_from, act_to), freq in dfg_counts.items():
        dep = dependency_measures.get((act_from, act_to), 0)
        
        if freq >= frequency_threshold and dep >= dependency_threshold:
            selected.append({
                'source': act_from,
                'target': act_to,
                'frequency': freq,
                'dependency': dep
            })
    
    return selected

# Create a sample event log
data = {
    'case:concept:name': ['A', 'A', 'A', 'B', 'B'],
    'concept:name':      ['Start', 'Process', 'End', 'Start', 'End'],
    'time:timestamp':    pd.to_datetime([
        '2024-01-01 10:00:00',
        '2024-01-01 10:05:00',
        '2024-01-01 10:10:00',
        '2024-01-01 10:00:00',
        '2024-01-01 10:15:00'
    ])
}

def discover_heuristics_net(df, dependency_threshold=0.5, frequency_threshold=1):
    """Discover a heuristics net from an event log DataFrame."""
    traces = extract_traces(df)
    activities = get_activity_set(traces)
    start_activities = get_start_activities(traces)
    end_activities = get_end_activities(traces)
    directly_follows_counts = compute_directly_follows(traces)
    dependency_measures = compute_dependency_measures(directly_follows_counts)
    selected_edges = select_edges(
        directly_follows_counts,
        dependency_measures,
        dependency_threshold=dependency_threshold,
        frequency_threshold=frequency_threshold
    )

    return {
        'traces': traces,
        'activities': activities,
        'start_activities': start_activities,
        'end_activities': end_activities,
        'directly_follows_counts': directly_follows_counts,
        'dependency_measures': dependency_measures,
        'selected_edges': selected_edges,
        'thresholds_used': {
            'dependency_threshold': dependency_threshold,
            'frequency_threshold': frequency_threshold
        }
    }

data = {
    'case:concept:name': ['A', 'A', 'A', 'B', 'B'],
    'concept:name':      ['Start', 'Process', 'End', 'Start', 'End'],
    'time:timestamp':    pd.to_datetime([
        '2024-01-01 10:00:00',
        '2024-01-01 10:05:00',
        '2024-01-01 10:10:00',
        '2024-01-01 10:00:00',
        '2024-01-01 10:15:00'
    ])
}

def save_model_summary(model, output_path):
    summary = {
        'number_of_activities': len(model['activities']),
        'number_of_selected_edges': len(model['selected_edges']),
        'start_activities': sorted(list(model['start_activities'])),
        'end_activities': sorted(list(model['end_activities'])),
        'selected_edges': model['selected_edges'],
        'thresholds_used': model['thresholds_used']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Example data
    data = {
        'case:concept:name': ['A', 'A', 'A', 'B', 'B'],
        'concept:name':      ['Start', 'Process', 'End', 'Start', 'End'],
        'time:timestamp':    pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 10:05:00',
            '2024-01-01 10:10:00',
            '2024-01-01 10:00:00',
            '2024-01-01 10:15:00'
        ])
    }
    df = pd.DataFrame(data)
    
    # Run and save
    result = discover_heuristics_net(df)
    save_model_summary(result, 'output/summary.json')
    
    print("Done!")
