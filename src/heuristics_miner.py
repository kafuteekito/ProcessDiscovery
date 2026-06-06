import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple


def extract_traces(df: pd.DataFrame) -> List[List[str]]:
    """Extract ordered activity sequences from event log DataFrame."""
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    if df.empty or not all(col in df.columns for col in required_cols):
        return []
    
    df_copy = df.copy()
    df_copy['time:timestamp'] = pd.to_datetime(df_copy['time:timestamp'])
    
    df_sorted = df_copy.sort_values(
        by=['case:concept:name', 'time:timestamp'],
        ascending=[True, True]
    )
    
    traces = []
    for case_id, group in df_sorted.groupby('case:concept:name'):
        activities = group['concept:name'].tolist()
        traces.append(activities)
    
    return traces


def get_start_activities(traces: List[List[str]]) -> Dict[str, int]:
    """Return dict of first activities with their counts."""
    if not traces:
        return {}
    start_activities = [trace[0] for trace in traces if trace]
    return dict(Counter(start_activities))


def get_end_activities(traces: List[List[str]]) -> Dict[str, int]:
    """Return dict of last activities with their counts."""
    if not traces:
        return {}
    end_activities = [trace[-1] for trace in traces if trace]
    return dict(Counter(end_activities))


def compute_directly_follows(traces: List[List[str]]) -> Tuple[Counter, Counter]:
    """Count directly follows relationships, separating normal pairs from self-loops."""
    normal_pairs = Counter()
    self_loops = Counter()
    
    for trace in traces:
        for i in range(len(trace) - 1):
            src, tgt = trace[i], trace[i + 1]
            if src == tgt:
                self_loops[src] += 1
            else:
                normal_pairs[(src, tgt)] += 1
    
    return normal_pairs, self_loops

def compute_dependency_measure(a: str, b: str, normal_pairs: Counter, self_loops: Counter) -> float:
    """Compute dependency measure between activities a and b."""
    if a == b:
        count = self_loops.get(a, 0)
        return count / (count + 1)
    else:
        count_ab = normal_pairs.get((a, b), 0)
        count_ba = normal_pairs.get((b, a), 0)
        return (count_ab - count_ba) / (count_ab + count_ba + 1)


def test_compute_dependency_measure():
    normal_pairs = Counter({('A', 'B'): 5, ('B', 'A'): 2, ('A', 'C'): 3})
    self_loops = Counter({'A': 4, 'B': 1})
    
    assert abs(compute_dependency_measure('A', 'B', normal_pairs, self_loops) - 0.375) < 0.001
    assert abs(compute_dependency_measure('A', 'A', normal_pairs, self_loops) - 0.8) < 0.001
    assert compute_dependency_measure('X', 'Y', normal_pairs, self_loops) == 0.0
    print("✓ test_compute_dependency_measure passed")

def filter_dependencies(normal_pairs: Counter, self_loops: Counter, 
                       dep_threshold: float = 0.5, freq_threshold: int = 2,
                       relative_to_best: float = 0.05) -> list:
    """Filter dependency pairs based on thresholds."""
    activities = set()
    for (a, b) in normal_pairs.keys():
        activities.add(a)
        activities.add(b)
    
    best_deps = {}
    for a in activities:
        max_dep = -1
        for b in activities:
            if (a, b) in normal_pairs or a == b:
                dep = compute_dependency_measure(a, b, normal_pairs, self_loops)
                max_dep = max(max_dep, dep)
        best_deps[a] = max_dep
    
    result = []
    for (a, b) in normal_pairs.keys():
        count = normal_pairs[(a, b)]
        if count < freq_threshold:
            continue
        
        dep_value = compute_dependency_measure(a, b, normal_pairs, self_loops)
        
        if dep_value < dep_threshold:
            continue
        
        if dep_value < best_deps.get(a, 0) - relative_to_best:
            continue
        
        result.append((a, b, dep_value))
    
    return result


def test_filter_dependencies():
    normal_pairs = Counter({('A', 'B'): 10, ('B', 'A'): 1})
    self_loops = Counter({'A': 0, 'B': 0})
    
    result = filter_dependencies(normal_pairs, self_loops, dep_threshold=0.3, freq_threshold=2, relative_to_best=0.1)
    
    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    assert result[0][0] == 'A' and result[0][1] == 'B', f"Expected (A, B, ...), got {result[0]}"
    assert abs(result[0][2] - 0.75) < 0.001, f"Expected dep ~0.75, got {result[0][2]}"
    print("✓ test_filter_dependencies passed")

def compute_and_probability(traces: list, a: str, b: str, c: str) -> float:
    """Compute probability that both b and c appear after last occurrence of a."""
    both = 0
    only_b = 0
    only_c = 0
    
    for trace in traces:
        if a not in trace:
            continue
        
        last_a_idx = len(trace) - 1 - trace[::-1].index(a)
        after_a = trace[last_a_idx + 1:]
        
        has_b = b in after_a
        has_c = c in after_a
        
        if has_b and has_c:
            both += 1
        elif has_b:
            only_b += 1
        elif has_c:
            only_c += 1
    
    denom = both + only_b + only_c
    return both / denom if denom > 0 else 0.0


def test_compute_and_probability():
    traces = [['A', 'B', 'C'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'A', 'C']]
    result = compute_and_probability(traces, 'A', 'B', 'C')
    assert abs(result - 0.25) < 0.001, f"Expected 0.25, got {result}"
    print("✓ test_compute_and_probability passed")


# Tests
if __name__ == "__main__":
    # Test extract_traces
    df = pd.DataFrame({
        'case:concept:name': ['A', 'A', 'B', 'B', 'C'],
        'concept:name': ['Start', 'End', 'Start', 'Process', 'Complete'],
        'time:timestamp': [2, 1, 1, 2, 1]
    })
    traces = extract_traces(df)
    assert traces == [['End', 'Start'], ['Start', 'Process'], ['Complete']], f"extract_traces failed: {traces}"
    print("✓ extract_traces passed")
    
    # Test get_start_activities
    traces = [['A', 'B', 'C'], ['A', 'D'], ['B', 'C'], ['A']]
    result = get_start_activities(traces)
    assert result == {'A': 3, 'B': 1}, f"get_start_activities failed: {result}"
    print("✓ get_start_activities passed")
    
    # Test get_end_activities
    traces = [['A', 'B', 'C'], ['A', 'D'], ['B', 'C'], ['E']]
    result = get_end_activities(traces)
    assert result == {'C': 2, 'D': 1, 'E': 1}, f"get_end_activities failed: {result}"
    print("✓ get_end_activities passed")
    
    # Test empty cases
    assert extract_traces(pd.DataFrame()) == [], "extract_traces empty case failed"
    assert get_start_activities([]) == {}, "get_start_activities empty case failed"
    assert get_end_activities([]) == {}, "get_end_activities empty case failed"
    print("✓ empty cases passed")
    
    # Test compute_directly_follows
    traces = [['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'A', 'C'], ['B', 'B']]
    normal, self_loops = compute_directly_follows(traces)
    assert normal == Counter({('A', 'B'): 2, ('B', 'C'): 1, ('A', 'C'): 1, ('B', 'D'): 1}), f"normal pairs failed: {normal}"
    assert self_loops == Counter({'A': 1, 'B': 1}), f"self loops failed: {self_loops}"
    print("✓ compute_directly_follows passed")
    
    print("\nAll tests passed!")

    test_compute_dependency_measure()

    test_filter_dependencies()

    test_compute_and_probability()

    