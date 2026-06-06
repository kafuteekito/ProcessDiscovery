import pandas as pd
from collections import Counter

def extract_traces(df):
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    
    if df.empty or not all(col in df.columns for col in required_cols):
        return []
    
    traces = []
    grouped = df.groupby('case:concept:name')
    
    for case_id, group in grouped:
        sorted_group = group.sort_values(by='time:timestamp', ascending=True)
        trace = sorted_group['concept:name'].tolist()
        traces.append(trace)
    
    return traces


def get_start_activities(traces):
    start_counts = {}
    
    for trace in traces:
        if trace:
            first_activity = trace[0]
            start_counts[first_activity] = start_counts.get(first_activity, 0) + 1
    
    return start_counts


def get_end_activities(traces):
    end_counts = {}
    
    for trace in traces:
        if trace:
            last_activity = trace[-1]
            end_counts[last_activity] = end_counts.get(last_activity, 0) + 1
    
    return end_counts


def compute_directly_follows(traces):
    normal_pairs = Counter()
    self_loops = Counter()
    
    for trace in traces:
        for i in range(len(trace) - 1):
            source, target = trace[i], trace[i + 1]
            if source == target:
                self_loops[source] += 1
            else:
                normal_pairs[(source, target)] += 1
    
    return normal_pairs, self_loops


def compute_dependency_measure(a, b, normal_pairs, self_loops):
    if a == b:
        count = self_loops.get(a, 0)
        return count / (count + 1)
    else:
        ab = normal_pairs.get((a, b), 0)
        ba = normal_pairs.get((b, a), 0)
        return (ab - ba) / (ab + ba + 1)
    
def get_best_outgoing(activity, normal_pairs, self_loops, activities):
    """Find the highest dependency measure among all outgoing edges from an activity."""
    if not activities:
        return 0.0
    
    max_dep = float('-inf')
    for target in activities:
        dep = compute_dependency_measure(activity, target, normal_pairs, self_loops)
        max_dep = max(max_dep, dep)
    
    return max_dep


if __name__ == "__main__":
    test_data = {
        'case:concept:name': ['Case1', 'Case1', 'Case2'],
        'concept:name': ['A', 'B', 'X'],
        'time:timestamp': [
            pd.Timestamp('2024-01-01 10:00:00'),
            pd.Timestamp('2024-01-01 09:00:00'),
            pd.Timestamp('2024-01-02 11:00:00'),
        ]
    }
    df = pd.DataFrame(test_data)
    assert extract_traces(df) == [['B', 'A'], ['X']]
    
    traces_3 = [['A', 'B', 'C'], ['A', 'D', 'E'], ['B', 'A', 'C']]
    assert get_start_activities(traces_3) == {'A': 2, 'B': 1}
    assert get_end_activities(traces_3) == {'C': 2, 'E': 1}
    
    traces_df = [['A', 'B', 'C'], ['A', 'A', 'B'], ['B', 'A', 'A']]
    normal_pairs, self_loops = compute_directly_follows(traces_df)
    assert normal_pairs == Counter({('A', 'B'): 2, ('B', 'C'): 1, ('B', 'A'): 1})
    assert self_loops == Counter({'A': 2})
    
    # Test compute_dependency_measure: 3 cases
    assert abs(compute_dependency_measure('A', 'A', normal_pairs, self_loops) - 2/3) < 0.001
    assert abs(compute_dependency_measure('A', 'B', normal_pairs, self_loops) - 1/4) < 0.001
    assert abs(compute_dependency_measure('B', 'A', normal_pairs, self_loops) - (-1/4)) < 0.001
    
    # Test with A→B=5, B→A=2: (5-2)/(5+2+1) = 3/8 = 0.375
    custom_pairs = Counter({('A', 'B'): 5, ('B', 'A'): 2})
    assert abs(compute_dependency_measure('A', 'B', custom_pairs, Counter()) - 0.375) < 0.001
    assert abs(compute_dependency_measure('B', 'A', custom_pairs, Counter()) - (-0.375)) < 0.001
    
    print("test passed")

    normal_pairs = Counter({('A', 'B'): 5, ('B', 'A'): 2, ('A', 'C'): 3})
    self_loops = Counter({'A': 1, 'B': 2})
    activities = ['A', 'B', 'C']
    
    # From A: A→A=1/2=0.5, A→B=(5-2)/(5+2+1)=0.375, A→C=(3-0)/(3+0+1)=0.75
    best_a = get_best_outgoing('A', normal_pairs, self_loops, activities)
    assert abs(best_a - 0.75) < 0.001
    
    # From B: B→A=(2-5)/(2+5+1)=-0.375, B→B=2/3=0.667, B→C=(0-0)/(0+0+1)=0
    best_b = get_best_outgoing('B', normal_pairs, self_loops, activities)
    assert abs(best_b - 2/3) < 0.001
    
    print("test passed")