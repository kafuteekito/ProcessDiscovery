from collections import Counter
from itertools import combinations, chain
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
import json
import os
from pm4py.visualization.petri_net import visualizer

def extract_traces(df):
    """
    Extract traces from an event log DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Event log with columns: 'case:concept:name', 'concept:name', 'time:timestamp'
    
    Returns:
    --------
    list
        List of tuples, where each tuple represents a case/trace as a sequence of activity names
    """
    traces = []
    
    for case_id, group in df.groupby('case:concept:name'):
        # Sort by timestamp using stable sort (mergesort) to preserve original order for ties
        sorted_group = group.sort_values('time:timestamp', kind='mergesort')
        
        # Extract concept names as a tuple
        trace = tuple(sorted_group['concept:name'])
        
        traces.append(trace)
    
    return traces

def get_activity_set(traces):
    """
    Extract all unique activity names from a list of traces.
    
    Parameters:
    -----------
    traces : list
        List of tuples, where each tuple represents a trace/case 
        containing ordered activity names
    
    Returns:
    --------
    set[str]
        Set of all unique activity names found across all traces
    """
    activities = set()
    
    for trace in traces:
        for activity in trace:
            activities.add(activity)
    
    return activities

def get_start_activities(traces):
    """
    Extract all unique start activities from a list of traces.
    
    Parameters:
    -----------
    traces : list
        List of tuples, where each tuple represents a trace/case 
        containing ordered activity names
    
    Returns:
    --------
    set[str]
        Set of all unique start activities (first activity of each non-empty trace)
    """
    start_activities = set()
    
    for trace in traces:
        if len(trace) > 0:
            start_activities.add(trace[0])
    
    return start_activities


def get_end_activities(traces):
    """
    Extract all unique end activities from a list of traces.
    
    Parameters:
    -----------
    traces : list
        List of tuples, where each tuple represents a trace/case 
        containing ordered activity names
    
    Returns:
    --------
    set[str]
        Set of all unique end activities (last activity of each non-empty trace)
    """
    end_activities = set()
    
    for trace in traces:
        if len(trace) > 0:
            end_activities.add(trace[-1])
    
    return end_activities

def get_directly_follows(traces):
    """
    Extract all unique directly-follows relationships from a list of traces.
    
    For each trace, collects all consecutive activity pairs (activity_a, activity_b)
    where activity_b directly follows activity_a.
    
    Parameters:
    -----------
    traces : list
        List of tuples, where each tuple represents a trace/case 
        containing ordered activity names
    
    Returns:
    --------
    set[tuple[str, str]]
        Set of unique (activity_a, activity_b) pairs representing directly-follows relations
    """
    directly_follows = set()
    
    for trace in traces:
        # Skip traces with fewer than 2 activities (no pairs possible)
        if len(trace) < 2:
            continue
        
        # Iterate through all consecutive pairs
        for i in range(len(trace) - 1):
            pair = (trace[i], trace[i + 1])
            directly_follows.add(pair)
    
    return directly_follows

def get_directly_follows_counts(traces):
    """
    Count occurrences of all directly-follows relationships from a list of traces.
    
    Parameters:
    -----------
    traces : list
        List of tuples, where each tuple represents a trace/case 
        containing ordered activity names
    
    Returns:
    --------
    Counter
        Counter object mapping (activity_a, activity_b) pairs to their occurrence counts
    """
    pairs = []
    
    for trace in traces:
        if len(trace) < 2:
            continue
        
        for i in range(len(trace) - 1):
            pairs.append((trace[i], trace[i + 1]))
    
    return Counter(pairs)

def get_footprint(activities, directly_follows):
    """
    Compute the footprint matrix for the Alpha Miner algorithm.
    
    For each ordered pair of activities (a, b), determines the relationship:
    - "→" : a directly follows b, but not vice versa (causal)
    - "←" : b directly follows a, but not vice versa (reverse causal)
    - "||" : both a->b and b->a exist (parallel)
    - "#" : neither direction exists (no relation)
    
    Parameters:
    -----------
    activities : set[str]
        Set of all unique activity names
    directly_follows : set[tuple[str, str]]
        Set of (activity_a, activity_b) pairs where b directly follows a
    
    Returns:
    --------
    dict[tuple[str, str], str]
        Dictionary mapping each ordered activity pair to its relationship symbol
    """
    footprint = {}
    
    for a in activities:
        for b in activities:
            a_follows_b = (a, b) in directly_follows
            b_follows_a = (b, a) in directly_follows
            
            if a_follows_b and not b_follows_a:
                footprint[(a, b)] = "→"
            elif not a_follows_b and b_follows_a:
                footprint[(a, b)] = "←"
            elif a_follows_b and b_follows_a:
                footprint[(a, b)] = "||"
            else:  # neither direction
                footprint[(a, b)] = "#"
    
    return footprint

def is_choice_coherent(subset, footprint):
    """
    Check if all activities in a subset are in "#" relation with each other.
    
    Parameters:
    -----------
    subset : set or frozenset
        Set of activity names to check
    footprint : dict[tuple[str, str], str]
        Footprint matrix mapping activity pairs to relationship symbols
    
    Returns:
    --------
    bool
        True if all pairs in subset have "#" relation (or subset is empty)
    """
    # Empty subset is trivially choice-coherent
    if len(subset) == 0:
        return True
    
    # Check all pairs within the subset (including when a1 == a2)
    for a1 in subset:
        for a2 in subset:
            if footprint[(a1, a2)] != "#":
                return False
    
    return True

def is_causally_connected(set_A, set_B, footprint):
    """
    Check if every activity in set_A causally leads to every activity in set_B.
    
    Parameters:
    -----------
    set_A : set or frozenset
        Source activities
    set_B : set or frozenset
        Target activities
    footprint : dict[tuple[str, str], str]
        Footprint matrix mapping activity pairs to relationship symbols
    
    Returns:
    --------
    bool
        True if all (a, b) pairs where a∈A, b∈B have "→" relation
        False if either set is empty
    """
    # Return False if either set is empty
    if len(set_A) == 0 or len(set_B) == 0:
        return False
    
    # Check all cross pairs
    for a in set_A:
        for b in set_B:
            if footprint[(a, b)] != "→":
                return False
    
    return True

def get_candidate_pairs(activities, footprint):
    """Generate all valid candidate pairs (A, B) for the Alpha Miner algorithm."""
    candidate_pairs = set()
    
    # Generate all non-empty subsets as frozensets
    activity_list = list(activities)
    all_subsets = [
        frozenset(combo)
        for r in range(1, len(activity_list) + 1)
        for combo in combinations(activity_list, r)
    ]
    
    # Filter to only choice-coherent subsets
    coherent_subsets = [S for S in all_subsets if is_choice_coherent(S, footprint)]
    
    # Check all pairs of coherent subsets
    for A in coherent_subsets:
        for B in coherent_subsets:
            if is_causally_connected(A, B, footprint):
                candidate_pairs.add((A, B))
    
    return candidate_pairs

def get_maximal_pairs(candidate_pairs):
    """
    Filter candidate pairs to keep only maximal pairs.
    
    A pair (A, B) is maximal if there is no other pair (A', B') in the set
    such that A ⊆ A' AND B ⊆ B' (where (A', B') ≠ (A, B)).
    
    Parameters:
    -----------
    candidate_pairs : set[tuple[frozenset[str], frozenset[str]]]
        Set of candidate (A, B) pairs
    
    Returns:
    --------
    set[tuple[frozenset[str], frozenset[str]]]
        Set of maximal pairs only
    """
    maximal_pairs = set()
    
    for pair in candidate_pairs:
        A, B = pair
        
        # Check if this pair is subsumed by any other pair
        is_subsumed = False
        
        for other_pair in candidate_pairs:
            # Explicitly skip comparing with itself
            if other_pair == pair:
                continue
            
            A_prime, B_prime = other_pair
            
            # Check if A ⊆ A' AND B ⊆ B'
            if A.issubset(A_prime) and B.issubset(B_prime):
                is_subsumed = True
                break
        
        # If not subsumed by any other pair, it's maximal
        if not is_subsumed:
            maximal_pairs.add(pair)
    
    return maximal_pairs

def build_petri_net(activities, start_activities, end_activities, maximal_pairs):
    """Build a Petri net using the Alpha Miner algorithm results."""
    net = PetriNet("Alpha Miner Net")
    
    # Create transitions for each activity
    transitions = {}
    for activity in activities:
        trans = PetriNet.Transition(activity, label=activity)
        transitions[activity] = trans
        net.transitions.add(trans)
    
    # Create source place (i_L) and sink place (o_L)
    source_place = PetriNet.Place("i_L")
    sink_place = PetriNet.Place("o_L")
    net.places.add(source_place)
    net.places.add(sink_place)
    
    # Create places for each maximal pair (A, B) and store mapping
    pair_places = {}
    for (A, B) in maximal_pairs:
        place_name = f"p_({','.join(sorted(A))},{','.join(sorted(B))})"
        place = PetriNet.Place(place_name)
        net.places.add(place)
        pair_places[(A, B)] = place
    
    # Type 1 & 2: Arcs between transitions and pair places
    for (A, B), place in pair_places.items():
        # Type 1: From each transition a in A to place p_(A,B)
        for a in A:
            add_arc_from_to(transitions[a], place, net)
        
        # Type 2: From place p_(A,B) to each transition b in B
        for b in B:
            add_arc_from_to(place, transitions[b], net)
    
    # Type 3: Arcs from source i_L to each start activity
    for start_act in start_activities:
        add_arc_from_to(source_place, transitions[start_act], net)
    
    # Type 4: Arcs from each end activity to sink o_L
    for end_act in end_activities:
        add_arc_from_to(transitions[end_act], sink_place, net)
    
    # Create initial marking (one token in source place i_L)
    initial_marking = Marking()
    initial_marking[source_place] = 1
    
    # Create final marking (one token in sink place o_L)
    final_marking = Marking()
    final_marking[sink_place] = 1
    
    return net, initial_marking, final_marking

def discover_alpha_net(df):
    """
    Discover a Petri net from an event log using the Alpha Miner algorithm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame with columns: 'case:concept:name', 'concept:name', 'time:timestamp'
    
    Returns:
    --------
    tuple[PetriNet, Marking, Marking]
        Petri net, initial marking, and final marking
    """
    # 1. Extract traces from DataFrame
    traces = extract_traces(df)
    
    # 2. Get set of activities
    activities = get_activity_set(traces)
    
    # 3. Get start and end activities
    start_activities = get_start_activities(traces)
    end_activities = get_end_activities(traces)
    
    # 4. Get directly-follows relations
    directly_follows = get_directly_follows(traces)
    
    # 5. Build footprint matrix
    footprint = get_footprint(activities, directly_follows)
    
    # 6. Generate candidate pairs
    candidate_pairs = get_candidate_pairs(activities, footprint)
    
    # 7. Filter to maximal pairs
    maximal_pairs = get_maximal_pairs(candidate_pairs)
    
    # 8. Build Petri net
    net, initial_marking, final_marking = build_petri_net(
        activities, 
        start_activities, 
        end_activities, 
        maximal_pairs
    )
    
    # 9. Return the Petri net tuple
    return net, initial_marking, final_marking

def save_alpha_summary(net, initial_marking, final_marking, output_path):
    """
    Save a JSON summary of the Alpha Miner Petri net model.
    
    Parameters:
    -----------
    net : PetriNet
        The discovered Petri net
    initial_marking : Marking
        Initial marking of the Petri net
    final_marking : Marking
        Final marking of the Petri net
    output_path : str
        Path to save the JSON file
    """
    # Extract transition labels
    transition_labels = [trans.label for trans in net.transitions if trans.label]
    
    # Extract place names
    place_names = [place.name for place in net.places if place.name]
    
    # Identify source place from initial marking
    source_place = None
    for place in initial_marking:
        source_place = place.name
        break
    
    # Identify sink place from final marking
    sink_place = None
    for place in final_marking:
        sink_place = place.name
        break
    
    # Build summary dictionary
    summary = {
        "num_places": len(net.places),
        "num_transitions": len(net.transitions),
        "num_arcs": len(net.arcs),
        "transition_labels": sorted(transition_labels),
        "place_names": sorted(place_names),
        "source_place": source_place,
        "sink_place": sink_place
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def visualize_alpha_net(net, initial_marking, final_marking, output_path):
    """
    Save a PNG visualization of the Alpha Miner Petri net.
    
    Parameters:
    -----------
    net : PetriNet
        The discovered Petri net
    initial_marking : Marking
        Initial marking of the Petri net
    final_marking : Marking
        Final marking of the Petri net
    output_path : str
        Path to save the PNG file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate visualization
    gviz = visualizer.apply(net, initial_marking, final_marking)
    
    # Save to PNG file
    visualizer.save(gviz, output_path)