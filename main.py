"""
Process Mining Module - Complete Toolkit

This module provides a complete toolkit for process mining with the Sepsis Cases event log:
- Event log loading and inspection
- Log statistics computation
- Process model discovery (Alpha, Heuristics, Inductive)
- Model quality evaluation (Fitness, Precision, Generalization, Simplicity)
- Multi-model comparison

All metrics use fast replay-based computation (not optimal alignments).
Compatible with pm4py 2.7.22.4+
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any, List

try:
    import pm4py
    from pm4py.objects.petri_net.obj import PetriNet
except ImportError:
    pm4py = None
    PetriNet = None


# =============================================================================
# SECTION 1: EVENT LOG LOADING
# =============================================================================

def load_event_log(file_path: str) -> pd.DataFrame:
    """
    Load an XES event log file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the .xes or .xes.gz event log file
    
    Returns:
    --------
    pd.DataFrame
        Event log as a pandas DataFrame with columns:
        - case:concept:name (case ID)
        - concept:name (activity name)
        - time:timestamp (event timestamp)
    
    Example:
    --------
    >>> df = load_event_log("data/Sepsis Cases - Event Log.xes.gz")
    >>> print(df.head())
    """
    if pm4py is None:
        raise ImportError("pm4py is required. Install with: pip install pm4py")
    
    # Read the XES log using pm4py (supports .xes and .xes.gz)
    event_log = pm4py.read_xes(file_path)
    
    # Convert the event log to a pandas DataFrame
    df = pm4py.convert_to_dataframe(event_log)
    
    return df


# =============================================================================
# SECTION 2: LOG STATISTICS
# =============================================================================

def compute_log_statistics(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive statistics from an event log DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame with columns:
        - case:concept:name (case ID)
        - concept:name (activity name)
        - time:timestamp (event timestamp)
    
    Returns:
    --------
    dict
        Dictionary containing various log statistics:
        - num_cases: Number of unique cases
        - num_events: Total number of events
        - num_activities: Number of unique activities
        - most_frequent_activities: Top 10 activities by frequency
        - start_activities: First activity per case (with counts)
        - end_activities: Last activity per case (with counts)
        - case_duration: Min/Max/Avg duration per case
    
    Example:
    --------
    >>> stats = compute_log_statistics(df)
    >>> print(f"Cases: {stats['num_cases']}, Events: {stats['num_events']}")
    """
    statistics = {}
    
    # -------------------------------------------------------------------------
    # Basic Counts
    # -------------------------------------------------------------------------
    
    # Count unique cases
    statistics['num_cases'] = df['case:concept:name'].nunique()
    
    # Count total events (rows)
    statistics['num_events'] = len(df)
    
    # Count unique activities
    statistics['num_activities'] = df['concept:name'].nunique()
    
    # -------------------------------------------------------------------------
    # Most Frequent Activities (Top 10)
    # -------------------------------------------------------------------------
    
    activity_counts = df['concept:name'].value_counts().head(10)
    statistics['most_frequent_activities'] = activity_counts.to_dict()
    
    # -------------------------------------------------------------------------
    # Start Activities (first activity per case)
    # -------------------------------------------------------------------------
    
    # Sort by case and timestamp, then get first activity per case
    df_sorted = df.sort_values(['case:concept:name', 'time:timestamp'])
    start_activities = df_sorted.groupby('case:concept:name')['concept:name'].first()
    start_activity_counts = start_activities.value_counts()
    statistics['start_activities'] = start_activity_counts.to_dict()
    
    # -------------------------------------------------------------------------
    # End Activities (last activity per case)
    # -------------------------------------------------------------------------
    
    # Get last activity per case
    end_activities = df_sorted.groupby('case:concept:name')['concept:name'].last()
    end_activity_counts = end_activities.value_counts()
    statistics['end_activities'] = end_activity_counts.to_dict()
    
    # -------------------------------------------------------------------------
    # Case Duration Statistics (min, max, average)
    # -------------------------------------------------------------------------
    
    # Calculate duration per case (max timestamp - min timestamp)
    case_durations = df_sorted.groupby('case:concept:name')['time:timestamp'].agg(
        lambda x: x.max() - x.min()
    )
    
    # Convert to seconds for easier interpretation
    case_durations_seconds = case_durations.dt.total_seconds()
    
    statistics['case_duration'] = {
        'min_seconds': case_durations_seconds.min(),
        'max_seconds': case_durations_seconds.max(),
        'avg_seconds': case_durations_seconds.mean(),
        'min_timedelta': case_durations.min(),
        'max_timedelta': case_durations.max(),
        'avg_timedelta': case_durations.mean()
    }
    
    return statistics


# =============================================================================
# SECTION 3: PROCESS MODEL DISCOVERY
# =============================================================================

def discover_alpha_net(df: pd.DataFrame) -> Optional[Tuple]:
    """
    Discover a Petri net using the Alpha algorithm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame
    
    Returns:
    --------
    tuple or None
        (PetriNet, initial_marking, final_marking) or None if failed
    """
    try:
        if pm4py is None:
            raise ImportError("pm4py is required")
        
        net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(df)
        return (net, initial_marking, final_marking)
    except Exception as e:
        print(f"Warning: Alpha net discovery failed - {str(e)}")
        return None


def discover_heuristics_net(df: pd.DataFrame) -> Optional[Tuple]:
    """
    Discover a Petri net using the Heuristics Miner algorithm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame
    
    Returns:
    --------
    tuple or None
        (PetriNet, initial_marking, final_marking) or None if failed
    """
    try:
        if pm4py is None:
            raise ImportError("pm4py is required")
        
        net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(df)
        return (net, initial_marking, final_marking)
    except Exception as e:
        print(f"Warning: Heuristics net discovery failed - {str(e)}")
        return None


def discover_inductive_net(df: pd.DataFrame) -> Optional[Tuple]:
    """
    Discover a Petri net using the Inductive Miner algorithm.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame (will be converted to event log internally)
    
    Returns:
    --------
    tuple or None
        (PetriNet, initial_marking, final_marking) or None if failed
    """
    try:
        if pm4py is None:
            raise ImportError("pm4py is required")
        
        # Inductive miner works better with event log object
        event_log = pm4py.format_dataframe(df)
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
        return (net, initial_marking, final_marking)
    except Exception as e:
        print(f"Warning: Inductive net discovery failed - {str(e)}")
        return None


# =============================================================================
# SECTION 4: MODEL QUALITY EVALUATION
# =============================================================================

def _validate_model(model: Tuple) -> bool:
    """
    Validate that model is in correct format (net, initial_marking, final_marking).
    
    Parameters:
    -----------
    model : tuple
        (PetriNet, initial_marking, final_marking)
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    if not isinstance(model, tuple) or len(model) != 3:
        return False
    
    net, initial_marking, final_marking = model
    
    # Basic validation
    if net is None:
        return False
    
    return True


def compute_fitness(model: Tuple, df: pd.DataFrame) -> Optional[float]:
    """
    Compute fitness score using replay-based token-based replay.
    
    Measures the proportion of observed behavior the model can reproduce.
    Uses fast replay-based computation (not optimal alignments).
    
    Parameters:
    -----------
    model : tuple
        (PetriNet, initial_marking, final_marking)
    df : pd.DataFrame
        Event log DataFrame
    
    Returns:
    --------
    float or None
        Fitness score between 0 and 1, or None if computation fails
    """
    try:
        if not _validate_model(model):
            return None
        
        net, initial_marking, final_marking = model
        
        # Use token-based replay (faster than alignments)
        fitness_result = pm4py.fitness_token_based_replay(
            df, 
            net, 
            initial_marking, 
            final_marking
        )
        
        # Extract fitness from result (returns dict with 'log_fitness' key for overall score)
        if isinstance(fitness_result, dict):
            return float(fitness_result.get('log_fitness', 0.0))
        else:
            return float(fitness_result)
            
    except Exception as e:
        print(f"Warning: Fitness computation failed - {str(e)}")
        return None


def compute_precision(model: Tuple, df: pd.DataFrame) -> Optional[float]:
    """
    Compute precision score measuring allowed-but-unseen behavior.
    
    Higher value means the model permits LESS unseen behavior (more precise).
    Uses token-based replay for faster computation.
    
    Parameters:
    -----------
    model : tuple
        (PetriNet, initial_marking, final_marking)
    df : pd.DataFrame
        Event log DataFrame
    
    Returns:
    --------
    float or None
        Precision score between 0 and 1, or None if computation fails
    """
    try:
        if not _validate_model(model):
            return None
        
        net, initial_marking, final_marking = model
        
        # Use token-based replay precision
        precision_result = pm4py.precision_token_based_replay(
            df, 
            net, 
            initial_marking, 
            final_marking
        )
        
        if isinstance(precision_result, dict):
            return float(precision_result.get('precision', 0.0))
        else:
            return float(precision_result)
            
    except Exception as e:
        print(f"Warning: Precision computation failed - {str(e)}")
        return None


def compute_generalization(model: Tuple, df: pd.DataFrame) -> Optional[float]:
    """
    Compute generalization score measuring coping with unseen behavior.
    
    Estimates how confidently the model can be trusted to allow further
    possible behavior, based on how often each model element is used.
    Higher means better generalization.
    
    Parameters:
    -----------
    model : tuple
        (PetriNet, initial_marking, final_marking)
    df : pd.DataFrame
        Event log DataFrame
    
    Returns:
    --------
    float or None
        Generalization score between 0 and 1, or None if computation fails
    """
    try:
        if not _validate_model(model):
            return None
        
        net, initial_marking, final_marking = model
        
        # Compute generalization using token-based replay
        generalization_result = pm4py.generalization_tbr(
            df, 
            net, 
            initial_marking, 
            final_marking
        )
        
        if isinstance(generalization_result, dict):
            return float(generalization_result.get('generalization', 0.0))
        else:
            return float(generalization_result)
            
    except Exception as e:
        print(f"Warning: Generalization computation failed - {str(e)}")
        return None


def compute_simplicity(model: Tuple, df: pd.DataFrame = None) -> Optional[float]:
    """
    Compute simplicity score measuring structural simplicity.
    
    Based on the number of incoming and outgoing connections of nodes.
    Higher means simpler structure. Does not require the log.
    
    Parameters:
    -----------
    model : tuple
        (PetriNet, initial_marking, final_marking)
    df : pd.DataFrame, optional
        Event log DataFrame (not used for simplicity, kept for API consistency)
    
    Returns:
    --------
    float or None
        Simplicity score between 0 and 1, or None if computation fails
    """
    try:
        if not _validate_model(model):
            return None
        
        net, initial_marking, final_marking = model
        
        # Compute simplicity using pm4py (arc-based)
        simplicity_result = pm4py.simplicity_petri_net(net, initial_marking, final_marking, variant='arc_degree')
        
        if isinstance(simplicity_result, dict):
            return float(simplicity_result.get('simplicity', 0.0))
        else:
            return float(simplicity_result)
            
    except Exception as e:
        print(f"Warning: Simplicity computation failed - {str(e)}")
        return None


def evaluate_model(model: Tuple, df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Evaluate a process model against an event log on all four quality dimensions.
    
    Parameters:
    -----------
    model : tuple
        Process model as (PetriNet, initial_marking, final_marking)
    df : pd.DataFrame
        Event log DataFrame with columns:
        - case:concept:name (case ID)
        - concept:name (activity name)
        - time:timestamp (event timestamp)
    
    Returns:
    --------
    dict
        Dictionary with keys: fitness, precision, generalization, simplicity
        Each value is a float between 0 and 1 (rounded to 4 decimals), or None if failed
    
    Example:
    --------
    >>> model = (net, initial_marking, final_marking)
    >>> results = evaluate_model(model, df)
    >>> print(results)
    {'fitness': 0.9234, 'precision': 0.8567, 'generalization': 0.7891, 'simplicity': 0.6543}
    """
    results = {}
    
    # Compute each metric independently (failures don't affect others)
    results['fitness'] = compute_fitness(model, df)
    results['precision'] = compute_precision(model, df)
    results['generalization'] = compute_generalization(model, df)
    results['simplicity'] = compute_simplicity(model, df)
    
    # Round all non-None values to 4 decimals
    for key in results:
        if results[key] is not None:
            results[key] = round(results[key], 4)
    
    return results


def compare_models(models_dict: Dict[str, Tuple], df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary mapping model names to model tuples
        e.g., {'alpha': model1, 'heuristics': model2, 'inductive': model3}
    df : pd.DataFrame
        Event log DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns
    """
    results = []
    
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, df)
        metrics['model'] = model_name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'fitness', 'precision', 'generalization', 'simplicity']
    comparison_df = comparison_df[[c for c in cols if c in comparison_df.columns]]
    
    return comparison_df


# =============================================================================
# SECTION 5: COMPLETE WORKFLOW EXAMPLE
# =============================================================================

def run_complete_analysis(file_path: str = "data/Sepsis Cases - Event Log.xes.gz"):
    """
    Run a complete process mining analysis pipeline.
    
    Parameters:
    -----------
    file_path : str
        Path to the XES event log file
    
    Returns:
    --------
    dict
        Dictionary containing:
        - df: Event log DataFrame
        - log_stats: Log statistics
        - models: Discovered models
        - evaluation: Model evaluation results
        - comparison: Model comparison DataFrame
    """
    print("=" * 70)
    print("PROCESS MINING COMPLETE ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Step 1: Load Event Log
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading event log...")
    try:
        df = load_event_log(file_path)
        results['df'] = df
        print(f"      ✓ Loaded {len(df):,} events")
        print(f"      ✓ {df['case:concept:name'].nunique():,} unique cases")
        print(f"      ✓ {df['concept:name'].nunique()} unique activities")
    except Exception as e:
        print(f"      ✗ Failed to load event log: {e}")
        return results
    
    # -------------------------------------------------------------------------
    # Step 2: Compute Log Statistics
    # -------------------------------------------------------------------------
    print("\n[2/5] Computing log statistics...")
    try:
        log_stats = compute_log_statistics(df)
        results['log_stats'] = log_stats
        print(f"      ✓ Cases: {log_stats['num_cases']:,}")
        print(f"      ✓ Events: {log_stats['num_events']:,}")
        print(f"      ✓ Activities: {log_stats['num_activities']}")
        print(f"      ✓ Avg case duration: {log_stats['case_duration']['avg_seconds']:,.2f} sec")
    except Exception as e:
        print(f"      ✗ Failed to compute statistics: {e}")
    
    # -------------------------------------------------------------------------
    # Step 3: Discover Process Models
    # -------------------------------------------------------------------------
    print("\n[3/5] Discovering process models...")
    models = {}
    
    # Alpha Miner
    try:
        model_alpha = discover_alpha_net(df)
        if model_alpha:
            models['alpha'] = model_alpha
            print("      ✓ Alpha miner completed")
        else:
            print("      ✗ Alpha miner returned None")
    except Exception as e:
        print(f"      ✗ Alpha miner failed: {e}")
    
    # Heuristics Miner
    try:
        model_heuristics = discover_heuristics_net(df)
        if model_heuristics:
            models['heuristics'] = model_heuristics
            print("      ✓ Heuristics miner completed")
        else:
            print("      ✗ Heuristics miner returned None")
    except Exception as e:
        print(f"      ✗ Heuristics miner failed: {e}")
    
    # Inductive Miner
    try:
        model_inductive = discover_inductive_net(df)
        if model_inductive:
            models['inductive'] = model_inductive
            print("      ✓ Inductive miner completed")
        else:
            print("      ✗ Inductive miner returned None")
    except Exception as e:
        print(f"      ✗ Inductive miner failed: {e}")
    
    results['models'] = models
    
    # -------------------------------------------------------------------------
    # Step 4: Evaluate Models
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating model quality...")
    if models:
        evaluation = {}
        for name, model in models.items():
            eval_result = evaluate_model(model, df)
            evaluation[name] = eval_result
            print(f"      ✓ {name}: fitness={eval_result['fitness']}, "
                  f"precision={eval_result['precision']}")
        results['evaluation'] = evaluation
    else:
        print("      ✗ No models to evaluate")
    
    # -------------------------------------------------------------------------
    # Step 5: Compare Models
    # -------------------------------------------------------------------------
    print("\n[5/5] Comparing models...")
    if models:
        comparison = compare_models(models, df)
        results['comparison'] = comparison
        print("\n" + comparison.to_string(index=False))
        
        # Best per metric
        print("\n      BEST MODEL PER METRIC:")
        for metric in ['fitness', 'precision', 'generalization', 'simplicity']:
            if metric in comparison.columns:
                valid_scores = comparison[metric].dropna()
                if len(valid_scores) > 0:
                    best_idx = valid_scores.idxmax()
                    best_model = comparison.loc[best_idx, 'model']
                    best_score = comparison.loc[best_idx, metric]
                    print(f"      {metric:15}: {best_model:12} ({best_score:.4f})")
    else:
        print("      ✗ No models to compare")
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Check dependencies
    if pm4py is None:
        print("ERROR: pm4py is not installed.")
        print("Install with: pip install pm4py")
        exit(1)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run complete analysis
    file_path = "data/Sepsis Cases - Event Log.xes.gz"
    results = run_complete_analysis(file_path)
    
    # Optionally save results
    if 'comparison' in results:
        results['comparison'].to_csv("results/metrics.csv", index=False)
        print("\n📁 Model comparison saved to: results/metrics.csv")
    
    if 'log_stats' in results:
        # Save log statistics as CSV (flattened structure)
        log_stats = results['log_stats']
        
        # Create a flat dictionary for CSV export
        stats_flat = {
            'num_cases': [log_stats['num_cases']],
            'num_events': [log_stats['num_events']],
            'num_activities': [log_stats['num_activities']],
            'case_duration_min_seconds': [log_stats['case_duration']['min_seconds']],
            'case_duration_max_seconds': [log_stats['case_duration']['max_seconds']],
            'case_duration_avg_seconds': [log_stats['case_duration']['avg_seconds']]
        }
        
        stats_df = pd.DataFrame(stats_flat)
        stats_df.to_csv("results/log_statistics.csv", index=False)
        print("📁 Log statistics saved to: results/log_statistics.csv")