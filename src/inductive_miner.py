from typing import Any, Dict, Set, Tuple
import json
import pandas as pd


def extract_start_end_activities(event_log: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """Extract start and end activities for each case from an event log."""
    start_activities: Set[str] = set()
    end_activities: Set[str] = set()
    
    for _, group in event_log.groupby('case:concept:name'):
        sorted_group = group.sort_values('time:timestamp')
        start_activities.add(sorted_group.iloc[0]['concept:name'])
        end_activities.add(sorted_group.iloc[-1]['concept:name'])
    
    return start_activities, end_activities


def compute_directly_follows_relations(event_log: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Compute directly follows relations from an event log."""
    directly_follows: Set[Tuple[str, str]] = set()
    
    for _, group in event_log.groupby('case:concept:name'):
        sorted_group = group.sort_values('time:timestamp')
        activities = sorted_group['concept:name'].tolist()
        
        for i in range(len(activities) - 1):
            directly_follows.add((activities[i], activities[i + 1]))
    
    return directly_follows


def build_adjacency_list(directly_follows: Set[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Build adjacency list from directly follows relations."""
    adjacency_list: Dict[str, Set[str]] = {}
    
    for source, target in directly_follows:
        if source not in adjacency_list:
            adjacency_list[source] = set()
        adjacency_list[source].add(target)
        
        if target not in adjacency_list:
            adjacency_list[target] = set()
    
    return adjacency_list


def compute_activity_neighbors(adjacency_list: Dict[str, Set[str]]) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """Compute predecessor and successor sets for all activities."""
    activity_info: Dict[str, Tuple[Set[str], Set[str]]] = {}
    
    all_activities: Set[str] = set(adjacency_list.keys())
    for targets in adjacency_list.values():
        all_activities.update(targets)
    
    for activity in all_activities:
        activity_info[activity] = (set(), set())
    
    for source, targets in adjacency_list.items():
        for target in targets:
            _, successors = activity_info[source]
            successors.add(target)
            
            predecessors, _ = activity_info[target]
            predecessors.add(source)
    
    return activity_info


def find_connected_components(adjacency_list: Dict[str, Set[str]]) -> list[Set[str]]:
    """Find connected components in an undirected graph."""
    visited: Set[str] = set()
    components: list[Set[str]] = []
    
    all_nodes: Set[str] = set(adjacency_list.keys())
    for targets in adjacency_list.values():
        all_nodes.update(targets)
    
    def dfs(node: str, component: Set[str]) -> None:
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        
        for neighbor in adjacency_list.get(node, set()):
            dfs(neighbor, component)
        
        for other_node, targets in adjacency_list.items():
            if node in targets and other_node not in visited:
                dfs(other_node, component)
    
    for node in all_nodes:
        if node not in visited:
            component: Set[str] = set()
            dfs(node, component)
            components.append(component)
    
    return components


def detect_xor_cut(adjacency_list: Dict[str, Set[str]]) -> list[Set[str]] | None:
    """Detect XOR cut by checking if graph has multiple connected components."""
    components = find_connected_components(adjacency_list)
    return components if len(components) > 1 else None


def detect_sequence_cut(adjacency_list: Dict[str, Set[str]]) -> list[Set[str]] | None:
    """Detect sequence cut by finding a valid topological split of activities."""
    all_activities: Set[str] = set(adjacency_list.keys())
    for targets in adjacency_list.values():
        all_activities.update(targets)
    
    if len(all_activities) <= 1:
        return None
    
    activity_neighbors = compute_activity_neighbors(adjacency_list)
    starts = {a for a in all_activities if len(activity_neighbors[a][0]) == 0}
    ends = {a for a in all_activities if len(activity_neighbors[a][1]) == 0}
    
    if not starts or not ends:
        return None
    
    ordered: list[Set[str]] = []
    remaining = set(all_activities)
    
    while remaining:
        current_layer = set()
        for activity in remaining:
            preds = activity_neighbors[activity][0]
            if preds.issubset(set().union(*ordered)):
                current_layer.add(activity)
        
        if not current_layer:
            return None
        
        ordered.append(current_layer)
        remaining -= current_layer
    
    return ordered if len(ordered) > 1 else None


def detect_parallel_cut(adjacency_list: Dict[str, Set[str]]) -> list[Set[str]] | None:
    """Detect parallel cut by finding bidirectional relationships between activities."""
    bidirectional_pairs: list[Set[str]] = []
    
    for source, targets in adjacency_list.items():
        for target in targets:
            # Check if reverse relationship exists (target -> source)
            if source in adjacency_list.get(target, set()):
                pair = frozenset([source, target])
                if pair not in [frozenset(p) for p in bidirectional_pairs]:
                    bidirectional_pairs.append({source, target})
    
    return bidirectional_pairs if bidirectional_pairs else None


def detect_loop_candidate(adjacency_list: Dict[str, Set[str]]) -> bool | None:
    """Detect loop candidate by checking for self-loops or cycles."""
    # Check for self-loops first
    for source, targets in adjacency_list.items():
        if source in targets:
            return True
    
    # Check for simple cycles using DFS
    def has_cycle(start: str, visited: Set[str], path: Set[str]) -> bool:
        if start in path:
            return True
        if start in visited:
            return False
        
        visited.add(start)
        path.add(start)
        
        for neighbor in adjacency_list.get(start, set()):
            if has_cycle(neighbor, visited, path):
                return True
        
        path.remove(start)
        return False
    
    visited: Set[str] = set()
    for node in adjacency_list.keys():
        if node not in visited:
            if has_cycle(node, visited, set()):
                return True
    
    return None


def discover_inductive_model(event_log: pd.DataFrame) -> Dict[str, Any]:
    """Discover process model from event log using inductive miner."""
    start_set, end_set = extract_start_end_activities(event_log)
    relations = compute_directly_follows_relations(event_log)
    adj_list = build_adjacency_list(relations)
    activity_neighbors = compute_activity_neighbors(adj_list)
    components = find_connected_components(adj_list)
    xor_cut = detect_xor_cut(adj_list)
    sequence_cut = detect_sequence_cut(adj_list)
    parallel_cut = detect_parallel_cut(adj_list)
    loop_candidate = detect_loop_candidate(adj_list)
    
    return {
        'start_activities': start_set,
        'end_activities': end_set,
        'relations': relations,
        'adjacency_list': adj_list,
        'activity_neighbors': activity_neighbors,
        'connected_components': components,
        'xor_cut': xor_cut,
        'sequence_cut': sequence_cut,
        'parallel_cut': parallel_cut,
        'loop_candidate': loop_candidate
    }


def summarize_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics for discovered model."""
    xor_cut = model.get('xor_cut')
    sequence_cut = model.get('sequence_cut')
    parallel_cut = model.get('parallel_cut')
    loop_candidate = model.get('loop_candidate')
    
    return {
        'algorithm': 'Inductive Miner',
        'num_start_activities': len(model['start_activities']),
        'num_end_activities': len(model['end_activities']),
        'num_relations': len(model['relations']),
        'num_activities': len(model['adjacency_list']),
        'num_components': len(model['connected_components']),
        'components': [list(c) for c in model['connected_components']],
        'implementation_status': 'simplified inductive miner with xor, sequence, parallel and loop candidate detection',
        'xor_cut_detected': xor_cut is not None,
        'sequence_cut_detected': sequence_cut is not None,
        'parallel_cut_detected': parallel_cut is not None,
        'loop_candidate_detected': loop_candidate is True,
        'num_xor_groups': len(xor_cut) if xor_cut else 0,
        'num_sequence_groups': len(sequence_cut) if sequence_cut else 0,
        'num_parallel_groups': len(parallel_cut) if parallel_cut else 0
    }


def save_model_summary(model: Dict[str, Any], output_path: str) -> None:
    """Save model summary to JSON file."""
    summary = summarize_model(model)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def make_activity_node(activity: str) -> Dict[str, Any]:
    """Create an activity leaf node."""
    return {'type': 'activity', 'name': activity}


def make_xor_node(*children: Dict[str, Any]) -> Dict[str, Any]:
    """Create an XOR (exclusive choice) operator node."""
    return {'type': 'xor', 'children': list(children)}


def make_sequence_node(*children: Dict[str, Any]) -> Dict[str, Any]:
    """Create a sequence (->) operator node."""
    return {'type': 'sequence', 'children': list(children)}


def make_parallel_node(*children: Dict[str, Any]) -> Dict[str, Any]:
    """Create a parallel (||) operator node."""
    return {'type': 'parallel', 'children': list(children)}


def make_flower_node(*children: Dict[str, Any]) -> Dict[str, Any]:
    """Create a flower (loop) operator node."""
    return {'type': 'flower', 'children': list(children)}


if __name__ == "__main__":
    # Create test event log with pm4py-style column names
    df = pd.DataFrame({
        'case:concept:name': ['Case1', 'Case1', 'Case1', 'Case2', 'Case2'],
        'concept:name': ['A', 'B', 'C', 'X', 'Y'],
        'time:timestamp': pd.to_datetime([
            '2024-01-01 09:00:00',
            '2024-01-01 10:00:00',
            '2024-01-01 11:00:00',
            '2024-01-01 09:30:00',
            '2024-01-01 10:30:00'
        ])
    })
    
    # Discover model
    model = discover_inductive_model(df)
    
    # Get summary
    summary = summarize_model(model)
    
    # Print summary
    print("Inductive Miner Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")