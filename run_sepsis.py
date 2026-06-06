import pandas as pd
import pm4py
import json
from src.heuristics_miner import (
    discover_heuristics_net,
    extract_traces,
    get_start_activities,
    get_end_activities,
    compute_directly_follows,
    filter_dependencies,
    compute_length2_loops
)

# Load event log
log = pm4py.read_xes('data/Sepsis Cases - Event Log.xes.gz')
df = pm4py.convert_to_dataframe(log)

# Extract traces for statistics
traces = extract_traces(df)
start_activities = get_start_activities(traces)
end_activities = get_end_activities(traces)
normal_pairs, self_loops = compute_directly_follows(traces)
length2_loops = compute_length2_loops(traces)
edges = filter_dependencies(normal_pairs, self_loops, dep_threshold=0.5, freq_threshold=2, 
                           relative_to_best=0.05, length2_loops=length2_loops, length2_threshold=0.9)

# Run heuristics miner with same parameters
net, initial_marking, final_marking = discover_heuristics_net(df, dep_threshold=0.5, freq_threshold=2, 
                                                              relative_to_best=0.05, length2_threshold=0.9)

# Print statistics
print(f"Places: {len(net.places)}")
print(f"Transitions: {len(net.transitions)}")
print(f"Arcs: {len(net.arcs)}")
print(f"Start activities: {len(start_activities)} ({list(start_activities.keys())})")
print(f"End activities: {len(end_activities)} ({list(end_activities.keys())})")
print(f"Filtered edges: {len(edges)}")

# Save results to JSON
results = {
    'places': len(net.places),
    'transitions': len(net.transitions),
    'arcs': len(net.arcs),
    'start_activities': list(start_activities.keys()),
    'end_activities': list(end_activities.keys()),
    'filtered_edges': len(edges),
    'edges': [{'from': a, 'to': b, 'dependency': d} for a, b, d in edges]
}

with open('results/heuristics_model_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results/heuristics_model_summary.json")