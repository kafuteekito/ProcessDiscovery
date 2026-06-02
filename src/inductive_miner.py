from typing import Dict, Set, Tuple, Any
import pandas as pd

import json

def extract_start_end_activities(event_log: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Extract start and end activities for each case from an event log.
    
    Parameters
    ----------
    event_log : pd.DataFrame
        DataFrame with 'case_id', 'activity', and 'timestamp' columns.
        
    Returns
    -------
    Tuple[Set[str], Set[str]]
        Tuple of (start_activities_set, end_activities_set).
    """
    start_activities: Set[str] = set()
    end_activities: Set[str] = set()
    
    for _, group in event_log.groupby('case_id'):
        sorted_group = group.sort_values('timestamp')
        start_activities.add(sorted_group.iloc[0]['activity'])
        end_activities.add(sorted_group.iloc[-1]['activity'])
    
    return start_activities, end_activities


def compute_directly_follows_relations(event_log: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Compute directly follows relations from an event log.
    
    Parameters
    ----------
    event_log : pd.DataFrame
        DataFrame with 'case_id', 'activity', and 'timestamp' columns.
        
    Returns
    -------
    Set[Tuple[str, str]]
        Set of (source_activity, target_activity) tuples.
    """
    directly_follows: Set[Tuple[str, str]] = set()
    
    for _, group in event_log.groupby('case_id'):
        sorted_group = group.sort_values('timestamp')
        activities = sorted_group['activity'].tolist()
        
        for i in range(len(activities) - 1):
            directly_follows.add((activities[i], activities[i + 1]))
    
    return directly_follows



def build_adjacency_list(directly_follows: Set[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """
    Build adjacency list from directly follows relations.
    
    Parameters
    ----------
    directly_follows : Set[Tuple[str, str]]
        Set of (source_activity, target_activity) tuples.
        
    Returns
    -------
    Dict[str, Set[str]]
        Dictionary mapping each activity to set of its successors.
    """
    adjacency_list: Dict[str, Set[str]] = {}
    
    for source, target in directly_follows:
        if source not in adjacency_list:
            adjacency_list[source] = set()
        adjacency_list[source].add(target)
        
        if target not in adjacency_list:
            adjacency_list[target] = set()
    
    return adjacency_list


def compute_activity_neighbors(adjacency_list: Dict[str, Set[str]]) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """
    Compute predecessor and successor sets for all activities.
    
    Parameters
    ----------
    adjacency_list : Dict[str, Set[str]]
        Adjacency list mapping activities to their direct successors.
        
    Returns
    -------
    Dict[str, Tuple[Set[str], Set[str]]]
        Dictionary mapping each activity to (predecessors_set, successors_set).
    """
    activity_info: Dict[str, Tuple[Set[str], Set[str]]] = {}
    
    # Collect all unique activities
    all_activities: Set[str] = set(adjacency_list.keys())
    for targets in adjacency_list.values():
        all_activities.update(targets)
    
    # Initialize all activities with empty sets
    for activity in all_activities:
        activity_info[activity] = (set(), set())
    
    # Populate predecessor and successor sets
    for source, targets in adjacency_list.items():
        for target in targets:
            _, successors = activity_info[source]
            successors.add(target)
            
            predecessors, _ = activity_info[target]
            predecessors.add(source)
    
    return activity_info


def find_connected_components(adjacency_list: Dict[str, Set[str]]) -> list[Set[str]]:
    """
    Find connected components in an undirected graph.
    
    Parameters
    ----------
    adjacency_list : Dict[str, Set[str]]
        Adjacency list representing directed edges. Treated as undirected.
        
    Returns
    -------
    list[Set[str]]
        List of sets, where each set contains activities in one connected component.
    """
    visited: Set[str] = set()
    components: list[Set[str]] = []
    
    # Build undirected view by collecting all nodes and their neighbors
    all_nodes: Set[str] = set(adjacency_list.keys())
    for targets in adjacency_list.values():
        all_nodes.update(targets)
    
    def dfs(node: str, component: Set[str]) -> None:
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        
        # Visit outgoing neighbors
        for neighbor in adjacency_list.get(node, set()):
            dfs(neighbor, component)
        
        # Visit incoming neighbors (treat as undirected)
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
    """
    Discover process model from event log using inductive miner.
    
    Parameters
    ----------
    event_log : pd.DataFrame
        Event log with 'case_id', 'activity', and 'timestamp' columns.
        
    Returns
    -------
    Dict[str, Any]
        Process model representation (incomplete implementation).
    """
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
    """
    Create summary statistics for discovered model.
    
    Parameters
    ----------
    model : Dict[str, Any]
        Process model from discover_inductive_model().
        
    Returns
    -------
    Dict[str, Any]
        Summary dictionary with model metrics.
    """
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
    """
    Save model summary to JSON file.
    
    Parameters
    ----------
    model : Dict[str, Any]
        Process model from discover_inductive_model().
    output_path : str
        Path to output JSON file.
    """
    summary = summarize_model(model)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)