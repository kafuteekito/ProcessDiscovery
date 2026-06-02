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


def discover_inductive_model(event_log: pd.DataFrame) -> Dict[str, Any]:
    """Discover process model from event log using inductive miner."""
    start_set, end_set = extract_start_end_activities(event_log)
    relations = compute_directly_follows_relations(event_log)
    adj_list = build_adjacency_list(relations)
    activity_neighbors = compute_activity_neighbors(adj_list)
    components = find_connected_components(adj_list)
    
    return {
        'start_activities': start_set,
        'end_activities': end_set,
        'relations': relations,
        'adjacency_list': adj_list,
        'activity_neighbors': activity_neighbors,
        'connected_components': components
    }


def summarize_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics for discovered model."""
    return {
        'algorithm': 'Inductive Miner',
        'num_start_activities': len(model['start_activities']),
        'num_end_activities': len(model['end_activities']),
        'num_relations': len(model['relations']),
        'num_activities': len(model['adjacency_list']),
        'num_components': len(model['connected_components']),
        'components': [list(c) for c in model['connected_components']],
        'implementation_status': 'partial graph-based inductive miner implementation'
    }


def save_model_summary(model: Dict[str, Any], output_path: str) -> None:
    """Save model summary to JSON file."""
    summary = summarize_model(model)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)