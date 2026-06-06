import pandas as pd
from collections import Counter
from itertools import combinations
from pm4py.objects.petri_net.obj import PetriNet

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

def filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05):
    """Filter dependency pairs based on frequency, dependency measure, and relative strength."""
    # Get all unique activities
    activities = set()
    for (a, b) in normal_pairs.keys():
        activities.add(a)
        activities.add(b)
    for a in self_loops.keys():
        activities.add(a)
    activities = list(activities)
    
    # Pre-compute best outgoing for each activity
    best_outgoing = {}
    for act in activities:
        best_outgoing[act] = get_best_outgoing(act, normal_pairs, self_loops, activities)
    
    filtered = []
    
    # Check normal pairs (a != b)
    for (a, b), count in normal_pairs.items():
        if count < freq_threshold:
            continue
        dep_value = compute_dependency_measure(a, b, normal_pairs, self_loops)
        if dep_value >= dep_threshold and dep_value >= best_outgoing[a] - relative_to_best:
            filtered.append((a, b, dep_value))
    
    # Check self-loops (a == b)
    for a, count in self_loops.items():
        if count < freq_threshold:
            continue
        dep_value = compute_dependency_measure(a, a, normal_pairs, self_loops)
        if dep_value >= dep_threshold and dep_value >= best_outgoing[a] - relative_to_best:
            filtered.append((a, a, dep_value))
    
    return filtered

def filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05):
    """Filter dependency pairs based on frequency, dependency measure, and relative strength."""
    activities = set()
    for (a, b) in normal_pairs.keys():
        activities.add(a)
        activities.add(b)
    for a in self_loops.keys():
        activities.add(a)
    activities = list(activities)
    
    best_outgoing = {}
    for act in activities:
        best_outgoing[act] = get_best_outgoing(act, normal_pairs, self_loops, activities)
    
    filtered = []
    
    for (a, b), count in normal_pairs.items():
        if count < freq_threshold:
            continue
        dep_value = compute_dependency_measure(a, b, normal_pairs, self_loops)
        if dep_value >= dep_threshold and dep_value >= best_outgoing[a] - relative_to_best:
            filtered.append((a, b, dep_value))
    
    for a, count in self_loops.items():
        if count < freq_threshold:
            continue
        dep_value = compute_dependency_measure(a, a, normal_pairs, self_loops)
        if dep_value >= dep_threshold and dep_value >= best_outgoing[a] - relative_to_best:
            filtered.append((a, a, dep_value))
    
    return filtered

def compute_and_probability(traces, a, b, c):
    """Compute probability that both b and c appear after last occurrence of a."""
    both = 0
    only_b = 0
    only_c = 0
    
    for trace in traces:
        if a not in trace:
            continue
        
        # Find index of last occurrence of a
        last_a_idx = len(trace) - 1 - trace[::-1].index(a)
        after_a = set(trace[last_a_idx + 1:])
        
        has_b = b in after_a
        has_c = c in after_a
        
        if has_b and has_c:
            both += 1
        elif has_b:
            only_b += 1
        elif has_c:
            only_c += 1
    
    total = both + only_b + only_c
    return both / total if total > 0 else 0.0

def get_split_candidates(edges):
    """Return dict mapping source activities to their targets, only for sources with 2+ outgoing edges."""
    from collections import defaultdict
    
    outgoing = defaultdict(list)
    for a, b, _ in edges:
        if b not in outgoing[a]:
            outgoing[a].append(b)
    
    return {a: targets for a, targets in outgoing.items() if len(targets) >= 2}

def detect_split_types(edges, traces, and_threshold=0.7, xor_threshold=0.3):
    """Detect split types (AND/XOR/OR) for each gateway activity."""
    split_candidates = get_split_candidates(edges)
    result = {}
    
    for activity, successors in split_candidates.items():
        if len(successors) < 2:
            continue
        
        pair_probs = []
        for b, c in combinations(successors, 2):
            prob = compute_and_probability(traces, activity, b, c)
            pair_probs.append(prob)
        
        avg_prob = sum(pair_probs) / len(pair_probs) if pair_probs else 0.0
        
        if avg_prob >= and_threshold:
            split_type = 'AND'
        elif avg_prob <= xor_threshold:
            split_type = 'XOR'
        else:
            split_type = 'OR'
        
        result[activity] = {
            'successors': successors,
            'type': split_type,
            'probability': avg_prob
        }
    
    return result

def create_transitions(activities, net):
    """Create one PetriNet.Transition per activity and add to net."""
    transitions = {}
    
    for activity in activities:
        trans = PetriNet.Transition(activity, activity)
        net.transitions.add(trans)
        transitions[activity] = trans
    
    return transitions

def create_edge_places(edges, transitions, net):
    """Create places and arcs for each edge in the dependency graph."""
    places = {}
    
    for a, b, _ in edges:
        place = PetriNet.Place(f"{a}_{b}")
        net.places.add(place)
        
        net.arcs.add(PetriNet.Arc(transitions[a], place))
        net.arcs.add(PetriNet.Arc(place, transitions[b]))
        
        places[(a, b)] = place
    
    return places

def add_source_place(net, transitions, start_activities):
    """Create source place with initial marking connected to start transitions."""
    source = PetriNet.Place("source")
    net.places.add(source)
    
    initial_marking = {source: 1}
    
    for activity in start_activities:
        if activity in transitions:
            net.arcs.add(PetriNet.Arc(source, transitions[activity]))
    
    return source, initial_marking

def add_sink_place(net, transitions, end_activities):
    """Create sink place with final marking connected from end transitions."""
    sink = PetriNet.Place("sink")
    net.places.add(sink)
    
    final_marking = {sink: 1}
    
    for activity in end_activities:
        if activity in transitions:
            net.arcs.add(PetriNet.Arc(transitions[activity], sink))
    
    return sink, final_marking

def create_transitions(activities, net):
    transitions = {}
    for activity in activities:
        trans = PetriNet.Transition(activity, activity)
        net.transitions.add(trans)
        transitions[activity] = trans
    return transitions


def create_edge_places(edges, transitions, net):
    places = {}
    for a, b, _ in edges:
        place = PetriNet.Place(f"{a}_{b}")
        net.places.add(place)
        net.arcs.add(PetriNet.Arc(transitions[a], place))
        net.arcs.add(PetriNet.Arc(place, transitions[b]))
        places[(a, b)] = place
    return places


def add_source_place(net, transitions, start_activities):
    source = PetriNet.Place("source")
    net.places.add(source)
    initial_marking = {source: 1}
    for activity in start_activities:
        if activity in transitions:
            net.arcs.add(PetriNet.Arc(source, transitions[activity]))
    return source, initial_marking


def add_sink_place(net, transitions, end_activities):
    sink = PetriNet.Place("sink")
    net.places.add(sink)
    final_marking = {sink: 1}
    for activity in end_activities:
        if activity in transitions:
            net.arcs.add(PetriNet.Arc(transitions[activity], sink))
    return sink, final_marking


def build_petri_net(edges, start_activities, end_activities):
    """Build a complete PetriNet from dependency edges."""
    net = PetriNet()
    
    activities = set()
    for a, b, _ in edges:
        activities.add(a)
        activities.add(b)
    activities = list(activities)
    
    transitions = create_transitions(activities, net)
    create_edge_places(edges, transitions, net)
    _, initial_marking = add_source_place(net, transitions, start_activities)
    _, final_marking = add_sink_place(net, transitions, end_activities)
    
    return net, initial_marking, final_marking





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

    normal_pairs = Counter({('A', 'B'): 5, ('B', 'A'): 2, ('A', 'C'): 3, ('C', 'D'): 1})
    self_loops = Counter({'A': 1, 'B': 3})
    
    # With dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05:
    # A→B: count=5>=2, dep=0.375<0.5 → excluded
    # B→A: count=2>=2, dep=-0.375<0.5 → excluded
    # A→C: count=3>=2, dep=0.75>=0.5, best_from_A=0.75, 0.75>=0.75-0.05 → included
    # C→D: count=1<2 → excluded
    # B→B: count=3>=2, dep=0.667>=0.5, best_from_B=0.667, 0.667>=0.667-0.05 → included
    
    result = filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05)
    
    assert len(result) == 2
    assert ('A', 'C', 0.75) in result
    
    print("test passed")

    normal_pairs = Counter({
        ('A', 'B'): 10, ('B', 'A'): 2,
        ('A', 'C'): 3, ('C', 'A'): 1,
        ('B', 'C'): 8, ('C', 'B'): 1,
        ('C', 'D'): 1
    })
    self_loops = Counter({'A': 5, 'B': 1})
    
    result = filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05)
    
    assert len(result) == 2
    
    found_bc = False
    found_aa = False
    for a, b, dep in result:
        if a == 'B' and b == 'C':
            found_bc = True
            assert abs(dep - 0.7) < 0.001
        elif a == 'A' and b == 'A':
            found_aa = True
            assert abs(dep - 5/6) < 0.001
    
    assert found_bc and found_aa
    
    print("test passed")

    normal_pairs = Counter({
        ('A', 'B'): 10, ('B', 'A'): 2,
        ('A', 'C'): 3, ('C', 'A'): 1,
        ('B', 'C'): 8, ('C', 'B'): 1,
        ('C', 'D'): 1
    })
    self_loops = Counter({'A': 5, 'B': 1})
    
    result = filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05)
    
    assert len(result) == 2
    
    dep_values = {(a, b): dep for a, b, dep in result}
    
    assert ('B', 'C') in dep_values
    assert abs(dep_values[('B', 'C')] - 0.7) < 0.001
    
    assert ('A', 'A') in dep_values
    assert abs(dep_values[('A', 'A')] - 5/6) < 0.001
    
    # Test with thresholds so high nothing passes
    result_empty = filter_dependencies(normal_pairs, self_loops, dep_threshold=0.99, freq_threshold=100, relative_to_best=0.0)
    assert result_empty == []
    
    print("test passed")

    traces = [
        ['A', 'B', 'C'],      # A then B,C → both
        ['A', 'B', 'D'],      # A then B only → only_b
        ['A', 'C', 'D'],      # A then C only → only_c
        ['A', 'B', 'C', 'A'], # last A has nothing after → excluded from counts
        ['X', 'A', 'B'],      # A then B only → only_b
        ['Y', 'Z'],           # no A → ignored
    ]
    
    # After last A: both=1, only_b=2, only_c=1
    # Result = 1 / (1 + 2 + 1) = 0.25
    prob = compute_and_probability(traces, 'A', 'B', 'C')
    assert abs(prob - 0.25) < 0.001
    
    print("test passed")

    prob_no_a = compute_and_probability(traces, 'X', 'B', 'C')
    assert prob_no_a == 0.0
    
    print("test passed")

    edges = [
        ('A', 'B', 0.8),
        ('A', 'C', 0.7),
        ('A', 'D', 0.6),
        ('B', 'C', 0.9),
        ('C', 'D', 0.5),
    ]
    
    result = get_split_candidates(edges)
    
    # A has 3 outgoing edges (B, C, D), B has 1, C has 1
    assert result == {'A': ['B', 'C', 'D']}
    
    # Test with no split candidates
    edges_single = [('A', 'B', 0.8), ('B', 'C', 0.9)]
    assert get_split_candidates(edges_single) == {}
    
    print("test passed")

    edges = [
        ('A', 'B', 0.8), ('A', 'C', 0.7), ('A', 'D', 0.6),
        ('X', 'Y', 0.9), ('X', 'Z', 0.85),
    ]
    
    # Traces where A always leads to all of B,C,D (AND split)
    # Traces where X leads to either Y or Z but never both (XOR split)
    traces = [
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'D', 'C'],
        ['X', 'Y'],
        ['X', 'Z'],
        ['X', 'Y'],
    ]
    
    result = detect_split_types(edges, traces, and_threshold=0.7, xor_threshold=0.3)
    
    assert 'A' in result
    assert result['A']['type'] == 'AND'
    assert result['A']['successors'] == ['B', 'C', 'D']
    assert abs(result['A']['probability'] - 1.0) < 0.001
    
    assert 'X' in result
    assert result['X']['type'] == 'XOR'
    assert result['X']['successors'] == ['Y', 'Z']
    assert abs(result['X']['probability'] - 0.0) < 0.001
    
    print("test passed")
    
        # Test AND-split where two activities almost always appear together
    edges_and = [
        ('P', 'Q', 0.85),
        ('P', 'R', 0.82),
    ]
    
    # 4 out of 5 traces have both Q and R after P (prob = 0.8)
    traces_and = [
        ['P', 'Q', 'R'],
        ['P', 'Q', 'R'],
        ['P', 'Q', 'R'],
        ['P', 'Q', 'R'],
        ['P', 'Q'],         # Only Q (1 out of 5)
    ]
    
    result_and = detect_split_types(edges_and, traces_and, and_threshold=0.7, xor_threshold=0.3)
    
    assert 'P' in result_and
    assert result_and['P']['type'] == 'AND'
    assert abs(result_and['P']['probability'] - 0.8) < 0.001
    
    print("test passed")

    net = PetriNet()
    activities = ['A', 'B', 'C', 'D']
    
    result = create_transitions(activities, net)
    
    assert len(result) == 4
    assert len(net.transitions) == 4
    
    for act in activities:
        assert act in result
        assert result[act].label == act
    
    print("test passed")

    net = PetriNet()
    activities = ['A', 'B', 'C']
    transitions = create_transitions(activities, net)
    
    edges = [('A', 'B', 0.8), ('A', 'C', 0.7), ('B', 'C', 0.9)]
    
    result = create_edge_places(edges, transitions, net)
    
    assert len(result) == 3
    assert len(net.places) == 3
    assert len(net.arcs) == 6  # 2 arcs per edge
    
    assert ('A', 'B') in result
    assert ('A', 'C') in result
    assert ('B', 'C') in result
    
    print("test passed")

    net = PetriNet()
    activities = ['A', 'B', 'C']
    transitions = create_transitions(activities, net)
    start_activities = ['A', 'B']
    
    source, initial_marking = add_source_place(net, transitions, start_activities)
    
    assert source in net.places
    assert initial_marking[source] == 1
    assert len(net.arcs) == 2  # One arc to each start transition
    
    print("test passed")

    net = PetriNet()
    activities = ['A', 'B', 'C']
    transitions = create_transitions(activities, net)
    end_activities = ['B', 'C']
    
    sink, final_marking = add_sink_place(net, transitions, end_activities)
    
    assert sink in net.places
    assert final_marking[sink] == 1
    assert len(net.arcs) == 2  # One arc from each end transition
    
    print("test passed")

    edges = [('A', 'B', 0.8), ('A', 'C', 0.7), ('B', 'D', 0.9), ('C', 'D', 0.85)]
    start_activities = ['A']
    end_activities = ['D']
    
    net, initial_marking, final_marking = build_petri_net(edges, start_activities, end_activities)
    
    assert len(net.transitions) == 4  # A, B, C, D
    assert len(net.places) == 6  # 4 edge places + source + sink
    assert len(net.arcs) == 10  # 8 edge arcs + 1 from source + 1 to sink
    
    print("test passed")