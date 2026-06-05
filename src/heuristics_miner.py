#!/usr/bin/env python3
"""process_miner.py - Process Mining Helper Functions (Heuristic Miner)"""

import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
from itertools import combinations
import json
from pm4py.objects.heuristics_net.obj import HeuristicsNet


def extract_traces(df: pd.DataFrame) -> List[List[str]]:
    """Extract traces from event log DataFrame grouped by case and sorted by timestamp."""
    traces = []
    for case_id, group in df.groupby('case:concept:name'):
        sorted_events = group.sort_values('time:timestamp')
        trace = sorted_events['concept:name'].tolist()
        traces.append(trace)
    return traces


def get_start_activities(traces: List[List[str]]) -> Dict[str, int]:
    """Get all activities that appear as the first activity in traces with counts."""
    start_activities = {}
    for trace in traces:
        if len(trace) > 0:
            start_act = trace[0]
            start_activities[start_act] = start_activities.get(start_act, 0) + 1
    return start_activities


def get_end_activities(traces: List[List[str]]) -> Dict[str, int]:
    """Get all activities that appear as the last activity in traces with counts."""
    end_activities = {}
    for trace in traces:
        if len(trace) > 0:
            end_act = trace[-1]
            end_activities[end_act] = end_activities.get(end_act, 0) + 1
    return end_activities


def compute_directly_follows_counts(df: pd.DataFrame) -> Tuple[Counter, Counter]:
    """Count directly-follows relations, returning normal follows and self-loops separately."""
    traces = extract_traces(df)
    normal_follows = Counter()
    self_loops = Counter()
    for trace in traces:
        for i in range(len(trace) - 1):
            activity_from = trace[i]
            activity_to = trace[i + 1]
            if activity_from == activity_to:
                self_loops[activity_from] += 1
            else:
                normal_follows[(activity_from, activity_to)] += 1
    return normal_follows, self_loops


def compute_dependency_measure(a: str, b: str, normal_follows: Counter, 
                                self_loops: Counter) -> float:
    """Compute dependency measure between two activities using Heuristic Miner formula."""
    if a == b:
        aa_count = self_loops.get(a, 0)
        return aa_count / (aa_count + 1)
    else:
        ab_count = normal_follows.get((a, b), 0)
        ba_count = normal_follows.get((b, a), 0)
        return (ab_count - ba_count) / (ab_count + ba_count + 1)


def filter_dependencies(normal_follows: Counter, self_loops: Counter,
                        dependency_threshold: float = 0.5,
                        frequency_threshold: int = 5,
                        all_dependencies_threshold: float = 0.05,
                        all_activities: bool = True) -> Dict:
    """Filter dependencies using three thresholds: dependency, frequency, and relative-to-best."""
    all_acts = set()
    for (a, b) in normal_follows.keys():
        all_acts.add(a)
        all_acts.add(b)
    for a in self_loops.keys():
        all_acts.add(a)
    
    edges = []
    loop_edges = []
    filtered_out = []
    
    best_per_activity = {}
    for act in all_acts:
        best_dep = -1
        for (a, b), count in normal_follows.items():
            if a == act:
                dep = compute_dependency_measure(a, b, normal_follows, self_loops)
                if dep > best_dep:
                    best_dep = dep
        best_per_activity[act] = best_dep
    
    for (a, b), count in normal_follows.items():
        dep_measure = compute_dependency_measure(a, b, normal_follows, self_loops)
        best_from_a = best_per_activity.get(a, 0)
        reasons = []
        
        if dep_measure < dependency_threshold:
            reasons.append(f"low_dependency ({dep_measure:.3f} < {dependency_threshold})")
        if count < frequency_threshold:
            reasons.append(f"low_frequency ({count} < {frequency_threshold})")
        relative_gap = best_from_a - dep_measure
        if relative_gap > all_dependencies_threshold:
            reasons.append(f"not_relative_to_best (gap={relative_gap:.3f} > {all_dependencies_threshold})")
        
        if reasons:
            filtered_out.append({
                'relation': (a, b), 'count': count, 'dependency': dep_measure,
                'best_from_source': best_from_a, 'relative_gap': relative_gap, 'reasons': reasons
            })
        else:
            edges.append((a, b, dep_measure))
    
    for a, count in self_loops.items():
        dep_measure = compute_dependency_measure(a, a, normal_follows, self_loops)
        best_from_a = best_per_activity.get(a, 0)
        reasons = []
        self_loop_threshold = dependency_threshold * 0.8
        if dep_measure < self_loop_threshold:
            reasons.append(f"low_dependency ({dep_measure:.3f} < {self_loop_threshold:.3f})")
        if count < frequency_threshold:
            reasons.append(f"low_frequency ({count} < {frequency_threshold})")
        relative_gap = best_from_a - dep_measure
        if relative_gap > all_dependencies_threshold:
            reasons.append(f"not_relative_to_best (gap={relative_gap:.3f} > {all_dependencies_threshold})")
        if reasons:
            filtered_out.append({
                'relation': (a, a), 'count': count, 'dependency': dep_measure,
                'best_from_source': best_from_a, 'relative_gap': relative_gap, 'reasons': reasons
            })
        else:
            loop_edges.append((a, dep_measure))
    
    activities_in_graph = set()
    for (a, b, _) in edges:
        activities_in_graph.add(a)
        activities_in_graph.add(b)
    for (a, _) in loop_edges:
        activities_in_graph.add(a)
    
    if all_activities:
        missing = all_acts - activities_in_graph
        for act in missing:
            best_edge = None
            best_dep = -1
            for (a, b), count in normal_follows.items():
                if a == act:
                    dep = compute_dependency_measure(a, b, normal_follows, self_loops)
                    if dep > best_dep:
                        best_dep = dep
                        best_edge = (a, b)
            best_in_edge = None
            best_in_dep = -1
            for (a, b), count in normal_follows.items():
                if b == act:
                    dep = compute_dependency_measure(a, b, normal_follows, self_loops)
                    if dep > best_in_dep:
                        best_in_dep = dep
                        best_in_edge = (a, b)
            if best_edge and (best_in_edge is None or best_dep >= best_in_dep):
                edges.append((best_edge[0], best_edge[1], best_dep))
            elif best_in_edge:
                edges.append((best_in_edge[0], best_in_edge[1], best_in_dep))
    
    return {
        'edges': edges, 'self_loops': loop_edges, 'filtered_out': filtered_out,
        'all_activities': all_acts, 'best_per_activity': best_per_activity
    }


def compute_and_probability(traces: List[List[str]], activity_a: str,
                            activity_b: str, activity_c: str) -> Tuple[float, Dict]:
    """Compute AND-probability for two successors after a common predecessor."""
    both_bc = 0
    only_b = 0
    only_c = 0
    neither = 0
    
    for trace in traces:
        a_positions = [i for i, act in enumerate(trace) if act == activity_a]
        if not a_positions:
            neither += 1
            continue
        last_a_pos = a_positions[-1]
        remainder = trace[last_a_pos + 1:]
        has_b = activity_b in remainder
        has_c = activity_c in remainder
        if has_b and has_c:
            both_bc += 1
        elif has_b:
            only_b += 1
        elif has_c:
            only_c += 1
        else:
            neither += 1
    
    total_with_successors = both_bc + only_b + only_c
    if total_with_successors == 0:
        and_prob = 0.0
    else:
        and_prob = both_bc / total_with_successors
    
    details = {
        'both_bc': both_bc, 'only_b': only_b, 'only_c': only_c,
        'neither': neither, 'total_with_successors': total_with_successors,
        'total_cases': len(traces)
    }
    return and_prob, details


def detect_split_types(edges: List[Tuple[str, str, float]], traces: List[List[str]],
                       and_threshold: float = 0.7, xor_threshold: float = 0.3) -> Dict:
    """Detect split types (AND/XOR/OR) for activities with multiple outgoing edges."""
    outgoing = {}
    for (a, b, dep) in edges:
        if a not in outgoing:
            outgoing[a] = []
        outgoing[a].append((b, dep))
    
    splits = {}
    summary = {'AND': 0, 'XOR': 0, 'OR': 0, 'activities_with_splits': 0}
    
    for activity, successors in outgoing.items():
        if len(successors) < 2:
            continue
        summary['activities_with_splits'] += 1
        splits[activity] = []
        successor_names = [s[0] for s in successors]
        for (succ_b, succ_c) in combinations(successor_names, 2):
            and_prob, details = compute_and_probability(traces, activity, succ_b, succ_c)
            if and_prob >= and_threshold:
                split_type = 'AND'
            elif and_prob <= xor_threshold:
                split_type = 'XOR'
            else:
                split_type = 'OR'
            summary[split_type] += 1
            splits[activity].append({
                'successors': (succ_b, succ_c), 'type': split_type,
                'probability': and_prob, 'details': details
            })
    
    return {'splits': splits, 'summary': summary}


def build_model_summary(edges, self_loops, start_activities, end_activities, 
                        split_result, thresholds, all_activities):
    """Build a simple summary dictionary of the discovered process model."""
    outgoing_edges = {}
    for (a, b, dep) in edges:
        if a not in outgoing_edges:
            outgoing_edges[a] = []
        outgoing_edges[a].append({'to': b, 'dependency': dep})
    
    incoming_edges = {}
    for (a, b, dep) in edges:
        if b not in incoming_edges:
            incoming_edges[b] = []
        incoming_edges[b].append({'from': a, 'dependency': dep})
    
    splits_simple = {}
    if split_result and 'splits' in split_result:
        for activity, split_list in split_result['splits'].items():
            splits_simple[activity] = []
            for s in split_list:
                splits_simple[activity].append({
                    'successors': list(s['successors']),
                    'type': s['type'],
                    'probability': round(s['probability'], 3)
                })
    
    gateways = list(splits_simple.keys())
    
    summary = {
        'metadata': {
            'total_activities': len(all_activities),
            'total_edges': len(edges),
            'total_self_loops': len(self_loops),
            'thresholds': thresholds
        },
        'activities': sorted(list(all_activities)),
        'start_activities': dict(start_activities),
        'end_activities': dict(end_activities),
        'edges': [
            {'from': a, 'to': b, 'dependency': round(dep, 3)}
            for (a, b, dep) in edges
        ],
        'self_loops': [
            {'activity': act, 'dependency': round(dep, 3)}
            for (act, dep) in self_loops
        ],
        'outgoing': outgoing_edges,
        'incoming': incoming_edges,
        'gateways': gateways,
        'splits': splits_simple,
        'split_summary': split_result.get('summary', {}) if split_result else {}
    }
    
    return summary


def print_model_summary(summary):
    """Print a human-readable summary of the process model."""
    print("\n" + "=" * 70)
    print("PROCESS MODEL SUMMARY")
    print("=" * 70)
    
    meta = summary['metadata']
    print(f"\nModel Statistics:")
    print(f"  Activities:     {meta['total_activities']}")
    print(f"  Edges:          {meta['total_edges']}")
    print(f"  Self-loops:     {meta['total_self_loops']}")
    
    print(f"\nThresholds Used:")
    for key, val in meta['thresholds'].items():
        print(f"  {key}: {val}")
    
    print(f"\nStart Activities: {list(summary['start_activities'].keys())}")
    print(f"End Activities:   {list(summary['end_activities'].keys())}")
    
    print(f"\nEdges ({len(summary['edges'])} total):")
    for edge in sorted(summary['edges'], key=lambda x: -x['dependency']):
        print(f"  {edge['from']} -> {edge['to']}: {edge['dependency']:.3f}")
    
    if summary['self_loops']:
        print(f"\nSelf-Loops ({len(summary['self_loops'])} total):")
        for loop in summary['self_loops']:
            print(f"  {loop['activity']} -> {loop['activity']}: {loop['dependency']:.3f}")
    
    if summary['gateways']:
        print(f"\nGateways ({len(summary['gateways'])} activities with splits):")
        for gw in summary['gateways']:
            print(f"  {gw}:")
            for split in summary['splits'].get(gw, []):
                succs = ' AND '.join(split['successors']) if split['type'] == 'AND' else ' XOR '.join(split['successors'])
                print(f"    [{split['type']}] {succs} (P={split['probability']:.3f})")
    
    print("\n" + "=" * 70)


def save_model_summary(summary, output_path):
    """Save process model summary as JSON file."""
    def convert_to_json_serializable(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    clean_summary = convert_to_json_serializable(summary)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_summary, f, indent=2, ensure_ascii=False)


def load_model_summary(input_path):
    """Load process model summary from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    return summary

def run_process_miner_on_dataframe(df):
    """Run complete process mining pipeline on provided DataFrame and save results."""
    import os
    
    os.makedirs('results', exist_ok=True)
    
    traces = extract_traces(df)
    starts = get_start_activities(traces)
    ends = get_end_activities(traces)
    normal_follows, self_loops = compute_directly_follows_counts(df)
    
    thresholds = {
        'dependency_threshold': 0.5,
        'frequency_threshold': 2,
        'all_dependencies_threshold': 0.05,
        'and_threshold': 0.7,
        'xor_threshold': 0.3
    }
    
    result = filter_dependencies(
        normal_follows,
        self_loops,
        dependency_threshold=thresholds['dependency_threshold'],
        frequency_threshold=thresholds['frequency_threshold'],
        all_dependencies_threshold=thresholds['all_dependencies_threshold'],
        all_activities=True
    )
    
    final_split_result = detect_split_types(
        result['edges'],
        traces,
        and_threshold=thresholds['and_threshold'],
        xor_threshold=thresholds['xor_threshold']
    )
    
    summary = build_model_summary(
        edges=result['edges'],
        self_loops=result['self_loops'],
        start_activities=starts,
        end_activities=ends,
        split_result=final_split_result,
        thresholds=thresholds,
        all_activities=result['all_activities']
    )
    
    print_model_summary(summary)
    save_model_summary(summary, 'results/heuristics_model_summary.json')
    print("\nModel saved to: results/heuristics_model_summary.json")
    
    return summary


def run_tests():
    """Run test cases for all functions."""
    print("=" * 70)
    print("PROCESS MINER - TEST SUITE")
    print("=" * 70)
    
    data = {
        'case:concept:name': ['Case1', 'Case1', 'Case1', 'Case2', 'Case2', 
                              'Case3', 'Case3', 'Case3', 'Case3', 'Case4', 
                              'Case4', 'Case4', 'Case5', 'Case5', 'Case5'],
        'concept:name':      ['A', 'B', 'C', 'A', 'C', 'A', 'B', 'D', 'E', 
                              'A', 'A', 'B', 'B', 'A', 'C'],
        'time:timestamp':    [1, 2, 3, 1, 2, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3]
    }
    df = pd.DataFrame(data)
    
    print("\nINPUT DATAFRAME")
    print("=" * 70)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TEST 1: extract_traces()")
    print("=" * 70)
    traces = extract_traces(df)
    print(f"\nExtracted {len(traces)} traces:")
    for i, trace in enumerate(traces, 1):
        print(f"  Trace {i}: {trace}")
    expected_traces = [['A', 'B', 'C'], ['A', 'C'], ['A', 'B', 'D', 'E'], 
                       ['A', 'A', 'B'], ['B', 'A', 'C']]
    print(f"\nExpected: {expected_traces}")
    test1 = traces == expected_traces
    print(f"Match: {test1}")
    
    print("\n" + "=" * 70)
    print("TEST 2: get_start_activities()")
    print("=" * 70)
    starts = get_start_activities(traces)
    print(f"\nStart activities: {starts}")
    expected_starts = {'A': 4, 'B': 1}
    print(f"Expected: {expected_starts}")
    test2 = starts == expected_starts
    print(f"Match: {test2}")
    
    print("\n" + "=" * 70)
    print("TEST 3: get_end_activities()")
    print("=" * 70)
    ends = get_end_activities(traces)
    print(f"\nEnd activities: {ends}")
    expected_ends = {'C': 3, 'E': 1, 'B': 1}
    print(f"Expected: {expected_ends}")
    test3 = ends == expected_ends
    print(f"Match: {test3}")
    
    print("\n" + "=" * 70)
    print("TEST 4: compute_directly_follows_counts()")
    print("=" * 70)
    normal_follows, self_loops = compute_directly_follows_counts(df)
    print(f"\nNormal follows:")
    for pair, count in sorted(normal_follows.items()):
        print(f"  {pair[0]} -> {pair[1]}: {count}")
    print(f"\nSelf-loops:")
    for activity, count in sorted(self_loops.items()):
        print(f"  {activity} -> {activity}: {count}")
    expected_normal = Counter({('A', 'B'): 3, ('B', 'C'): 1, ('A', 'C'): 2, 
                               ('B', 'D'): 1, ('D', 'E'): 1, ('B', 'A'): 1})
    expected_self_loops = Counter({'A': 1})
    print(f"\nExpected normal: {dict(expected_normal)}")
    print(f"Match: {normal_follows == expected_normal}")
    print(f"\nExpected self-loops: {dict(expected_self_loops)}")
    test4 = normal_follows == expected_normal and self_loops == expected_self_loops
    print(f"Match: {test4}")
    
    print("\n" + "=" * 70)
    print("TEST 5: compute_dependency_measure()")
    print("=" * 70)
    print("\nDependency Measures:")
    print("-" * 50)
    dep_ab = compute_dependency_measure('A', 'B', normal_follows, self_loops)
    expected_ab = 2 / 5
    print(f"  A -> B: {dep_ab:.4f} (expected: {expected_ab:.4f})")
    dep_ba = compute_dependency_measure('B', 'A', normal_follows, self_loops)
    expected_ba = -2 / 5
    print(f"  B -> A: {dep_ba:.4f} (expected: {expected_ba:.4f})")
    dep_aa = compute_dependency_measure('A', 'A', normal_follows, self_loops)
    expected_aa = 1 / 2
    print(f"  A -> A: {dep_aa:.4f} (expected: {expected_aa:.4f})")
    dep_ac = compute_dependency_measure('A', 'C', normal_follows, self_loops)
    expected_ac = 2 / 3
    print(f"  A -> C: {dep_ac:.4f} (expected: {expected_ac:.4f})")
    test5 = (abs(dep_ab - expected_ab) < 0.0001 and abs(dep_ba - expected_ba) < 0.0001 and
             abs(dep_aa - expected_aa) < 0.0001 and abs(dep_ac - expected_ac) < 0.0001)
    print(f"\nAll measures correct: {test5}")
    
    print("\n" + "=" * 70)
    print("TEST 6: filter_dependencies()")
    print("=" * 70)
    result = filter_dependencies(normal_follows, self_loops, dependency_threshold=0.3,
                                  frequency_threshold=1, all_dependencies_threshold=0.3,
                                  all_activities=True)
    print(f"\nKept {len(result['edges'])} normal edges:")
    for (a, b, dep) in sorted(result['edges'], key=lambda x: -x[2]):
        print(f"  {a} -> {b}: {dep:.3f}")
    print(f"\nKept {len(result['self_loops'])} self-loops:")
    for (a, dep) in sorted(result['self_loops'], key=lambda x: -x[1]):
        print(f"  {a} -> {a}: {dep:.3f}")
    print(f"\nFiltered out {len(result['filtered_out'])} relations")
    print(f"\nAll activities: {result['all_activities']}")
    acts_in_graph = set()
    for (a, b, _) in result['edges']:
        acts_in_graph.add(a)
        acts_in_graph.add(b)
    for (a, _) in result['self_loops']:
        acts_in_graph.add(a)
    test6 = result['all_activities'].issubset(acts_in_graph)
    print(f"All activities covered: {test6}")
    
    print("\n" + "=" * 70)
    print("TEST 7: Split Type Detection (AND vs XOR)")
    print("=" * 70)
    split_data = {
        'case:concept:name': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3',
                              'C4', 'C4', 'C5', 'C5', 'C6', 'C6', 'C6', 'C7', 'C7'],
        'concept:name':      ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'C', 'B',
                              'A', 'B', 'A', 'C', 'A', 'B', 'D', 'A', 'C'],
        'time:timestamp':    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2]
    }
    split_df = pd.DataFrame(split_data)
    split_traces = extract_traces(split_df)
    print("\nTest Traces:")
    for i, trace in enumerate(split_traces, 1):
        print(f"  Trace {i}: {' -> '.join(trace)}")
    
    print("\n" + "-" * 70)
    print("TEST 7a: compute_and_probability('A', 'B', 'C')")
    print("-" * 70)
    and_prob, details = compute_and_probability(split_traces, 'A', 'B', 'C')
    print(f"\nAnalysis of B and C after A:")
    print(f"  Cases with BOTH B and C: {details['both_bc']}")
    print(f"  Cases with only B:       {details['only_b']}")
    print(f"  Cases with only C:       {details['only_c']}")
    print(f"  Cases with neither:      {details['neither']}")
    print(f"  Total with successors:   {details['total_with_successors']}")
    print(f"\n  AND-probability: {and_prob:.3f}")
    expected_both = 3
    expected_only_b = 2
    expected_only_c = 2
    expected_total = 7
    expected_prob = 3 / 7
    print(f"\nExpected: both={expected_both}, only_b={expected_only_b}, "
          f"only_c={expected_only_c}, total={expected_total}")
    print(f"Expected P_AND: {expected_prob:.3f}")
    test7a = (details['both_bc'] == expected_both and details['only_b'] == expected_only_b and
              details['only_c'] == expected_only_c and 
              details['total_with_successors'] == expected_total and
              abs(and_prob - expected_prob) < 0.001)
    print(f"Match: {test7a}")
    
    print("\n" + "-" * 70)
    print("TEST 7b: detect_split_types()")
    print("-" * 70)
    test_edges = [('A', 'B', 0.6), ('A', 'C', 0.5), ('A', 'D', 0.4), 
                  ('B', 'E', 0.8), ('C', 'E', 0.7)]
    split_result = detect_split_types(test_edges, split_traces, 
                                       and_threshold=0.7, xor_threshold=0.3)
    
    print("\n" + "=" * 70)
    print("GENERATING FINAL MODEL OUTPUT")
    print("=" * 70)
    
    thresholds = {
        'dependency_threshold': 0.3,
        'frequency_threshold': 1,
        'all_dependencies_threshold': 0.3,
        'and_threshold': 0.7,
        'xor_threshold': 0.3
    }
    
    # Detect splits on the main test traces (not TEST 7's separate split_traces)
    final_split_result = detect_split_types(
        result['edges'], 
        traces,
        and_threshold=thresholds['and_threshold'],
        xor_threshold=thresholds['xor_threshold']
    )
    
    summary = build_model_summary(
        edges=result['edges'],
        self_loops=result['self_loops'],
        start_activities=starts,
        end_activities=ends,
        split_result=final_split_result,
        thresholds=thresholds,
        all_activities=result['all_activities']
    )
    
    print_model_summary(summary)
    save_model_summary(summary, 'model_summary.json')
    print("\nModel saved to: model_summary.json")
    
    test7b = 'splits' in split_result and 'summary' in split_result
    print(f"\nResult structure correct: {test7b}")
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    all_passed = test1 and test2 and test3 and test4 and test5 and test6 and test7a and test7b
    if all_passed:
        print("\n[OK] ALL TESTS PASSED!")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        results = [('extract_traces', test1), ('get_start_activities', test2),
                   ('get_end_activities', test3), ('compute_directly_follows_counts', test4),
                   ('compute_dependency_measure', test5), ('filter_dependencies', test6),
                   ('compute_and_probability', test7a), ('detect_split_types', test7b)]
        for name, passed in results:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {name}")
    print("\n" + "=" * 70)
    return all_passed


if __name__ == "__main__":
    run_tests()
