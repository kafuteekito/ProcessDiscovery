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
    from collections import Counter
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
    max_dep = float('-inf')
    for target in activities:
        dep = compute_dependency_measure(activity, target, normal_pairs, self_loops)
        max_dep = max(max_dep, dep)
    return max_dep


def filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05):
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


def detect_split_types(edges, traces, and_threshold=0.7, xor_threshold=0.3):
    from itertools import combinations
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
        result[activity] = {'successors': successors, 'type': split_type, 'probability': avg_prob}
    return result


def get_split_candidates(edges):
    from collections import defaultdict
    outgoing = defaultdict(list)
    for a, b, _ in edges:
        if b not in outgoing[a]:
            outgoing[a].append(b)
    return {a: targets for a, targets in outgoing.items() if len(targets) >= 2}


def compute_and_probability(traces, a, b, c):
    both = only_b = only_c = 0
    for trace in traces:
        if a not in trace:
            continue
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


def discover_heuristics_net(df, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05):
    """Discover a PetriNet using the heuristics miner algorithm."""
    traces = extract_traces(df)
    
    if not traces:
        net = PetriNet()
        return net, {}, {}
    
    start_activities = list(get_start_activities(traces).keys())
    end_activities = list(get_end_activities(traces).keys())
    
    normal_pairs, self_loops = compute_directly_follows(traces)
    
    edges = filter_dependencies(normal_pairs, self_loops, dep_threshold, freq_threshold, relative_to_best)
    
    detect_split_types(edges, traces)
    
    net, initial_marking, final_marking = build_petri_net(edges, start_activities, end_activities)
    
    return net, initial_marking, final_marking


# Test discover_heuristics_net
if __name__ == "__main__":
    import pandas as pd
    
    df = pd.DataFrame({
        'case:concept:name': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3'],
        'concept:name': ['A', 'B', 'D', 'A', 'C', 'D', 'A', 'B', 'D'],
        'time:timestamp': pd.to_datetime([
            '2024-01-01 09:00', '2024-01-01 10:00', '2024-01-01 11:00',
            '2024-01-01 09:00', '2024-01-01 10:00', '2024-01-01 11:00',
            '2024-01-01 09:00', '2024-01-01 10:00', '2024-01-01 11:00',
        ])
    })
    
    net, initial_marking, final_marking = discover_heuristics_net(df, dep_threshold=0.5, freq_threshold=2, relative_to_best=0.05)
    
    assert len(net.transitions) >= 3  # At least A, B, D
    assert len(net.places) >= 4  # Edge places + source + sink
    assert len(initial_marking) == 1
    assert len(final_marking) == 1
    
    print("test passed")