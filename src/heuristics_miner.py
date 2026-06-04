#!/usr/bin/env python3
"""
Heuristic Miner Pipeline for Process Discovery

This script implements the Heuristic Miner algorithm based on:
- Weijters et al. (2006): "Discovering Process Nets with Frequency Labels"
- Enhanced with AND/XOR split detection

Usage:
    python heuristic_miner_pipeline.py <path_to_xes_file>
    
Example:
    python heuristic_miner_pipeline.py sepsis_cases.xes
"""

import argparse
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking


# =============================================================================
# STEP 1: Load Event Log
# =============================================================================

def load_event_log(filepath):
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Event log not found: {filepath}")
    
    xes_path = str(filepath)
    
    # Load XES (returns EventLog, not DataFrame)
    event_log = pm4py.read_xes(xes_path)
    
    # Convert EventLog to DataFrame
    df = pm4py.convert_to_dataframe(event_log)
    
    # Rename to standardized columns
    column_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'case' in col_lower:
            column_map[col] = 'case:concept:name'
        elif col == 'concept:name' or 'concept:name' in col_lower:
            column_map[col] = 'concept:name'
        elif 'time' in col_lower or 'timestamp' in col_lower:
            column_map[col] = 'time:timestamp'
    
    if column_map:
        df = df.rename(columns=column_map)
    
    # Ensure required columns exist
    required = ['case:concept:name', 'concept:name', 'time:timestamp']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    df = df.sort_values(['case:concept:name', 'time:timestamp'])
    
    return df


# =============================================================================
# STEP 2: Compute Succession Counts
# =============================================================================

def compute_direct_succession_counts(df):
    """
    Compute direct succession and self-loop counts.
    
    Returns:
        tuple: (direct_counts, self_loop_counts)
    """
    direct_counts = Counter()
    self_loop_counts = Counter()
    
    for case_id, group in df.groupby('case:concept:name', sort=False):
        activities = group['concept:name'].tolist()
        for i in range(len(activities) - 1):
            a, b = activities[i], activities[i + 1]
            if a == b:
                self_loop_counts[(a, a)] += 1
            else:
                direct_counts[(a, b)] += 1
    
    return direct_counts, self_loop_counts


def compute_triple_counts(df):
    """
    Compute length-3 succession counts for AND-split detection.
    
    Returns:
        Counter: mapping (A, B, C) → count
    """
    triple_counts = Counter()
    
    for case_id, group in df.groupby('case:concept:name', sort=False):
        activities = group['concept:name'].tolist()
        for i in range(len(activities) - 2):
            triple = (activities[i], activities[i + 1], activities[i + 2])
            triple_counts[triple] += 1
    
    return triple_counts


# =============================================================================
# STEP 3: Compute Dependency Matrix
# =============================================================================

def compute_dependency_matrix(df, smooth=1):
    """
    Compute the full dependency matrix.
    
    Includes:
    - Length-2 dependencies: C(A,B) = (|AB| - |BA|) / (|AB| + |BA| + 1)
    - Length-1 self-loops: C(A,A) = |AA| / (|AA| + 1)
    
    Parameters:
        df: Event log DataFrame
        smooth: Laplace smoothing parameter (default 1)
    
    Returns:
        dict with matrix, counts, and metadata
    """
    direct_counts, self_loop_counts = compute_direct_succession_counts(df)
    activities = list(df['concept:name'].unique())
    
    matrix = {}
    
    # Length-2 dependencies
    for a, b in product(activities, activities):
        if a == b:
            continue
        
        ab = direct_counts.get((a, b), 0)
        ba = direct_counts.get((b, a), 0)
        denominator = ab + ba + smooth
        matrix[(a, b)] = (ab - ba) / denominator if denominator > 0 else 0.0
    
    # Length-1 self-loops
    for a in activities:
        aa = self_loop_counts.get((a, a), 0)
        matrix[(a, a)] = aa / (aa + smooth)
    
    return {
        'matrix': matrix,
        'direct_counts': dict(direct_counts),
        'self_loop_counts': dict(self_loop_counts),
        'activities': activities,
        'smooth': smooth
    }


def compute_max_output_dependencies(dep_matrix):
    """
    Compute the maximum dependency value for each activity's outgoing edges.
    
    Used for relative threshold calculation.
    """
    matrix = dep_matrix['matrix']
    activities = dep_matrix['activities']
    
    max_output = {}
    for a in activities:
        max_c = 0.0
        for b in activities:
            if a != b:
                c = matrix.get((a, b), 0.0)
                if c > max_c:
                    max_c = c
        max_output[a] = max_c
    
    return max_output


# =============================================================================
# STEP 4: Filter Strong Dependencies
# =============================================================================

def filter_strong_dependencies(dep_matrix, threshold_dependency=0.5,
                                threshold_frequency=1, threshold_relative=0.5):
    """
    Filter dependencies using three threshold conditions.
    
    Strong dependency A → B iff:
        1. (|AB| + |BA|) >= threshold_frequency
        2. |C(A,B)| >= threshold_dependency
        3. C(A,B) >= threshold_relative × max_output(A)
    
    Parameters:
        dep_matrix: Dependency matrix from compute_dependency_matrix()
        threshold_dependency: Minimum |C(A,B)| value
        threshold_frequency: Minimum (|AB| + |BA|) count
        threshold_relative: Minimum fraction of max output (0=disabled)
    
    Returns:
        dict: {(A, B): info_dict}
    """
    matrix = dep_matrix['matrix']
    direct_counts = dep_matrix['direct_counts']
    max_output = compute_max_output_dependencies(dep_matrix)
    
    strong = {}
    
    for (a, b), c_value in matrix.items():
        if a == b:
            continue
        
        ab = direct_counts.get((a, b), 0)
        ba = direct_counts.get((b, a), 0)
        total = ab + ba
        
        passes_freq = total >= threshold_frequency
        passes_abs = abs(c_value) >= threshold_dependency
        max_c = max_output.get(a, 0.0)
        passes_rel = c_value >= threshold_relative * max_c
        
        if passes_freq and passes_abs and passes_rel:
            strong[(a, b)] = {
                'c': c_value,
                'ab': ab,
                'ba': ba,
                'total': total,
                'relative_to_max': c_value / max_c if max_c > 0 else 0.0
            }
    
    return strong


# =============================================================================
# STEP 5: Detect AND/XOR Splits
# =============================================================================

def compute_and_likelihood(df, dep_matrix, smooth=1):
    """
    Compute AND-likelihood for all activity triples.
    
    L(A,B,C) = (|ABC| + |ACB|) / (|AB| + |AC| + 1)
    
    High value → B and C tend to co-occur after A (AND-split)
    Low value → B and C tend to NOT co-occur (XOR-split)
    """
    direct_counts = dep_matrix['direct_counts']
    triple_counts = compute_triple_counts(df)
    activities = dep_matrix['activities']
    
    and_likelihood = {}
    
    for a in activities:
        for b in activities:
            if b == a:
                continue
            for c in activities:
                if c == a or c == b:
                    continue
                
                abc = triple_counts.get((a, b, c), 0)
                acb = triple_counts.get((a, c, b), 0)
                ab = direct_counts.get((a, b), 0)
                ac = direct_counts.get((a, c), 0)
                
                denominator = ab + ac + smooth
                l = (abc + acb) / denominator if denominator > 0 else 0.0
                
                and_likelihood[(a, b, c)] = l
    
    return and_likelihood


def detect_splits(df, dep_matrix, threshold_dependency=0.5,
                  threshold_frequency=1, threshold_relative=0.5,
                  threshold_and=0.5):
    """
    Detect XOR-splits and AND-splits in the process.
    
    Parameters:
        df: Event log DataFrame
        dep_matrix: Dependency matrix
        threshold_and: Minimum AND-likelihood for AND-split (below = XOR)
    
    Returns:
        dict with xor_splits, and_splits, outgoing, and_likelihood
    """
    smooth = 1
    direct_counts = dep_matrix['direct_counts']
    matrix = dep_matrix['matrix']
    activities = dep_matrix['activities']
    max_output = compute_max_output_dependencies(dep_matrix)
    and_likelihood = compute_and_likelihood(df, dep_matrix)
    
    # Find strong outgoing edges
    outgoing = {}
    
    for a in activities:
        outgoing[a] = []
        max_c = max_output.get(a, 0.0)
        
        for b in activities:
            if a != b:
                c = matrix.get((a, b), 0.0)
                ab = direct_counts.get((a, b), 0)
                ba = direct_counts.get((b, a), 0)
                total = ab + ba
                
                passes_dep = abs(c) >= threshold_dependency
                passes_freq = total >= threshold_frequency
                passes_rel = c >= threshold_relative * max_c
                
                if passes_dep and passes_freq and passes_rel:
                    outgoing[a].append((b, c, total))
    
    # Sort by dependency strength
    for a in outgoing:
        outgoing[a].sort(key=lambda x: -x[1])
    
    # Detect split types
    xor_splits = []
    and_splits = []
    
    for a, targets in outgoing.items():
        if len(targets) < 2:
            continue
        
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                b, c = targets[i][0], targets[j][0]
                
                l = and_likelihood.get((a, b, c), and_likelihood.get((a, c, b), 0.0))
                
                split_info = {
                    'source': a,
                    'targets': [b, c],
                    'and_likelihood': l
                }
                
                if l >= threshold_and:
                    and_splits.append(split_info)
                else:
                    xor_splits.append(split_info)
    
    return {
        'xor_splits': xor_splits,
        'and_splits': and_splits,
        'outgoing': outgoing,
        'and_likelihood': and_likelihood
    }


# =============================================================================
# STEP 6: Build Petri Net
# =============================================================================

def build_petri_net(df, dep_matrix, splits):
    """
    Build a Petri net from dependency graph and split detection.
    """
    print("[DEBUG] build_petri_net: starting...")
    
    activities = dep_matrix['activities']
    matrix = dep_matrix['matrix']
    outgoing = splits['outgoing']
    xor_splits = splits['xor_splits']
    and_splits = splits['and_splits']
    
    print(f"[DEBUG] activities: {len(activities)}, outgoing keys: {list(outgoing.keys())}")
    print(f"[DEBUG] xor_splits: {xor_splits}")
    print(f"[DEBUG] and_splits: {and_splits}")
    
    net = PetriNet("Heuristic Miner Petri Net")
    
    # Create transitions
    transitions = {}
    for activity in activities:
        t = PetriNet.Transition(activity, activity)
        transitions[activity] = t
        net.transitions.add(t)
    
    print(f"[DEBUG] created {len(transitions)} transitions")
    
    def create_merge_transition(name):
        t = PetriNet.Transition(f"merge_{name}", None)
        transitions[f"merge_{name}"] = t
        net.transitions.add(t)
        return t
    
    places = {}
    
    def get_or_create_place(name):
        if name not in places:
            places[name] = PetriNet.Place(name)
            net.places.add(places[name])
        return places[name]
    
    def add_arc(source, target):
        net.arcs.add(PetriNet.Arc(source, target))
    
    # Identify source and sink activities
    has_incoming = defaultdict(bool)
    has_outgoing = defaultdict(bool)
    
    for (a, b), c in matrix.items():
        if a != b:
            has_outgoing[a] = True
            has_incoming[b] = True
    
    source_activities = [a for a in activities if not has_incoming[a]]
    sink_activities = [a for a in activities if not has_outgoing[a]]
    
    print(f"[DEBUG] source_activities: {source_activities}, sink_activities: {sink_activities}")
    
    source_place = get_or_create_place("source")
    sink_place = get_or_create_place("sink")
    
    for activity in source_activities:
        add_arc(source_place, transitions[activity])
    
    for activity in sink_activities:
        add_arc(transitions[activity], sink_place)
    
    pending_connections = defaultdict(list)
    
    for a, targets in outgoing.items():
        if not targets:
            print(f"[DEBUG] No targets for {a}")
            continue
        
        place_after_a = get_or_create_place(f"p_{a}")
        add_arc(transitions[a], place_after_a)
        
        if len(targets) == 1:
            b = targets[0][0]
            add_arc(place_after_a, transitions[b])
        
        elif len(targets) == 2:
            b, c = targets[0][0], targets[1][0]
            
            is_and = any(
                s['source'] == a and set(s['targets']) == {b, c}
                for s in and_splits
            )
            
            if is_and:
                place_b = get_or_create_place(f"p_{a}_{b}")
                place_c = get_or_create_place(f"p_{a}_{c}")
                
                add_arc(place_after_a, transitions[b])
                add_arc(place_after_a, transitions[c])
                add_arc(transitions[b], place_b)
                add_arc(transitions[c], place_c)
                
                pending_connections[f"join_{b}_{c}"] = [(place_b, 1), (place_c, 1)]
            
            else:
                for target in targets:
                    t = target[0]
                    place_t = get_or_create_place(f"p_{a}_{t}")
                    add_arc(place_after_a, transitions[t])
                    add_arc(transitions[t], place_t)
                    pending_connections[f"xor_merge_{a}"].append(place_t)
    
    print(f"[DEBUG] pending_connections: {dict(pending_connections)}")
    
    for merge_key, source_places in pending_connections.items():
        if len(source_places) == 1:
            add_arc(source_places[0], sink_place)
        else:
            merge_trans = create_merge_transition(merge_key)
            
            for src_place in source_places:
                add_arc(src_place, merge_trans)
            
            add_arc(merge_trans, sink_place)
    
    im = Marking({source_place: 1})
    fm = Marking({sink_place: 1})
    
    print(f"[DEBUG] build_petri_net: complete. places={len(net.places)}, transitions={len(net.transitions)}, arcs={len(net.arcs)}")
    
    return net, im, fm


# =============================================================================
# STEP 7: Analysis and Reporting
# =============================================================================

def print_dependency_summary(dep_matrix, strong_deps):
    """Print summary of dependency analysis."""
    print("\n" + "=" * 80)
    print("DEPENDENCY ANALYSIS SUMMARY")
    print("=" * 80)
    
    activities = dep_matrix['activities']
    matrix = dep_matrix['matrix']
    direct_counts = dep_matrix['direct_counts']
    
    print(f"\nTotal unique activities: {len(activities)}")
    print(f"Total direct succession pairs: {len(direct_counts)}")
    print(f"Strong dependencies: {len(strong_deps)}")
    
    # Top dependencies by strength
    print("\n--- Top 10 Strongest Dependencies ---")
    sorted_deps = sorted(strong_deps.items(), key=lambda x: -x[1]['c'])
    for (a, b), info in sorted_deps[:10]:
        direction = "->" if info['c'] > 0 else "<-"
        print(f"  {a} {direction} {b}: C={info['c']:+.3f}, "
              f"AB={info['ab']}, BA={info['ba']}")
    
    # Self-loops
    print("\n--- Self-Loop Activities (C(A,A) > 0.5) ---")
    self_loops = [(a, c) for (a, a), c in matrix.items() if a == a and c > 0.5]
    if self_loops:
        for a, c in sorted(self_loops, key=lambda x: -x[1]):
            print(f"  {a}: C={c:.3f}")
    else:
        print("  (none detected)")


def print_split_summary(splits):
    """Print summary of split detection."""
    print("\n" + "=" * 80)
    print("SPLIT DETECTION SUMMARY")
    print("=" * 80)
    
    xor_splits = splits['xor_splits']
    and_splits = splits['and_splits']
    
    print(f"\nXOR-splits detected: {len(xor_splits)}")
    print(f"AND-splits detected: {len(and_splits)}")
    
    if xor_splits:
        print("\n--- XOR-Splits (Exclusive Choice) ---")
        for s in xor_splits:
            print(f"  {s['source']} -> [{s['targets'][0]} OR {s['targets'][1]}] "
                  f"(L={s['and_likelihood']:.3f})")
    
    if and_splits:
        print("\n--- AND-Splits (Parallel Execution) ---")
        for s in and_splits:
            print(f"  {s['source']} -> [{s['targets'][0]} AND {s['targets'][1]}] "
                  f"(L={s['and_likelihood']:.3f})")


def print_petri_net_summary(net, im, fm):
    """Print summary of the resulting Petri net."""
    print("\n" + "=" * 80)
    print("PETRI NET SUMMARY")
    print("=" * 80)
    
    print(f"\nNet name: {net.name}")
    print(f"Places: {len(net.places)}")
    print(f"Transitions: {len(net.transitions)}")
    print(f"Arcs: {len(net.arcs)}")
    
    print(f"\nInitial Marking: {dict(im)}")
    print(f"Final Marking: {dict(fm)}")
    
    print("\n--- Transitions ---")
    for t in sorted(net.transitions, key=lambda x: x.label or ""):
        label = t.label or "(silent)"
        print(f"  ( {label} )")
    
    print("\n--- Places ---")
    for p in sorted(net.places, key=lambda x: x.name):
        # Count incoming and outgoing arcs
        in_arcs = len([a for a in net.arcs if a.target == p])
        out_arcs = len([a for a in net.arcs if a.source == p])
        print(f"  [ {p.name} ]  (in: {in_arcs}, out: {out_arcs})")
    
    print("\n--- All Arcs ---")
    for arc in sorted(net.arcs, key=lambda x: (
        x.source.name if hasattr(x.source, 'name') else (x.source.label or ""),
        x.target.name if hasattr(x.target, 'name') else (x.target.label or "")
    )):
        source_name = arc.source.name if hasattr(arc.source, 'name') else (arc.source.label or "source")
        target_name = arc.target.name if hasattr(arc.target, 'name') else (arc.target.label or "target")
        print(f"  {source_name} -> {target_name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(filepath, 
                  threshold_dependency=0.5,
                  threshold_frequency=1,
                  threshold_relative=0.5,
                  threshold_and=0.5,
                  visualize=True,
                  export_format='pnml'):
    """
    Run the complete Heuristic Miner pipeline.
    
    Parameters:
        filepath: Path to event log file
        threshold_dependency: Dependency threshold
        threshold_frequency: Frequency threshold
        threshold_relative: Relative threshold
        threshold_and: AND-likelihood threshold
        visualize: Whether to visualize the Petri net
        export_format: Format to export ('pnml', 'png', None)
    
    Returns:
        tuple: (net, im, fm)
    """
    print("=" * 80)
    print("HEURISTIC MINER PIPELINE")
    print("=" * 80)
    
    print(f"\nLoading event log from: {filepath}")
    df = load_event_log(filepath)
    print(f"Loaded {len(df)} events across {df['case:concept:name'].nunique()} cases")
    
    # Step 1: Compute dependency matrix
    print("\n[Step 1/4] Computing dependency matrix...")
    dep_matrix = compute_dependency_matrix(df)
    print(f"  Found {len(dep_matrix['activities'])} activities")
    
    # Step 2: Filter strong dependencies
    print("\n[Step 2/4] Filtering strong dependencies...")
    print(f"  thresholds: dep={threshold_dependency}, freq={threshold_frequency}, rel={threshold_relative}")
    strong_deps = filter_strong_dependencies(
        dep_matrix,
        threshold_dependency=threshold_dependency,
        threshold_frequency=threshold_frequency,
        threshold_relative=threshold_relative
    )
    print(f"  Found {len(strong_deps)} strong dependencies")
    
    # Step 3: Detect splits
    print("\n[Step 3/4] Detecting XOR/AND splits...")
    print(f"  threshold_and={threshold_and}")
    splits = detect_splits(
        df, dep_matrix,
        threshold_dependency=threshold_dependency,
        threshold_frequency=threshold_frequency,
        threshold_relative=threshold_relative,
        threshold_and=threshold_and
    )
    print(f"  Found {len(splits['xor_splits'])} XOR-splits, {len(splits['and_splits'])} AND-splits")
    
    # Step 4: Build Petri net
    print("\n[Step 4/4] Building Petri net...")
    net, im, fm = build_petri_net(df, dep_matrix, splits)
    print(f"  Created {len(net.places)} places, {len(net.transitions)} transitions, "
          f"{len(net.arcs)} arcs")
    
    # Print summaries
    print_dependency_summary(dep_matrix, strong_deps)
    print_split_summary(splits)
    print_petri_net_summary(net, im, fm)
    
    # Export
    if export_format:
        output_path = Path(filepath).stem
        if export_format == 'pnml':
            output_file = f"{output_path}_petri_net.pnml"
            pm4py.write_pnml(net, im, fm, output_file)
            print(f"\nPetri net exported to: {output_file}")
        elif export_format == 'png' and visualize:
            output_file = f"{output_path}_petri_net.png"
            try:
                gviz = pm4py.visualization.petri_net.apply(net, im, fm)
                pm4py.visualization.petri_net.save(gviz, output_file)
                print(f"Petri net visualization saved to: {output_file}")
            except Exception as e:
                print(f"Could not save visualization: {e}")
    
    # Visualize
    if visualize:
        try:
            print("\n[Visualizing Petri net...]")
            gviz = pm4py.visualization.petri_net.apply(net, im, fm)
            pm4py.visualization.petri_net.view(gviz)
        except Exception as e:
            print(f"Could not visualize: {e}")
    
    return net, im, fm


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Heuristic Miner Pipeline for Process Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python heuristic_miner_pipeline.py sepsis_cases.xes
  python heuristic_miner_pipeline.py log.xes --dep 0.7 --freq 2 --rel 0.5 --and-thresh 0.6
  python heuristic_miner_pipeline.py log.csv --no-visualize
        """
    )
    
    parser.add_argument('filepath', help="Path to XES or CSV event log file")
    parser.add_argument('--dep', type=float, default=0.5,
                       help="Dependency threshold (default: 0.5)")
    parser.add_argument('--freq', type=int, default=1,
                       help="Frequency threshold (default: 1)")
    parser.add_argument('--rel', type=float, default=0.5,
                       help="Relative threshold (default: 0.5, use 0 to disable)")
    parser.add_argument('--and-thresh', type=float, default=0.5,
                       help="AND-likelihood threshold (default: 0.5)")
    parser.add_argument('--no-visualize', action='store_true',
                       help="Disable visualization")
    parser.add_argument('--export', choices=['pnml', 'png', 'none'], 
                       default='pnml', help="Export format (default: pnml)")
    
    args = parser.parse_args()
    
    export_format = args.export if args.export != 'none' else None
    
    try:
        run_pipeline(
            filepath=args.filepath,
            threshold_dependency=args.dep,
            threshold_frequency=args.freq,
            threshold_relative=args.rel,
            threshold_and=args.and_thresh,  # FIXED: was args.and
            visualize=not args.no_visualize,
            export_format=export_format
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTo download the Sepsis Cases event log:", file=sys.stderr)
        print("  1. Visit: https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12677382", 
              file=sys.stderr)
        print("  2. Download the XES file", file=sys.stderr)
        print("  3. Save it as 'sepsis_cases.xes' in the current directory", 
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()