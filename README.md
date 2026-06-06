# ProcessDiscovery

A process mining toolkit that discovers and compares process models from real-world event logs. Built as a university project using the **Sepsis Cases** healthcare dataset.

Three process discovery algorithms are implemented from scratch and evaluated side-by-side on four quality metrics: fitness, precision, generalization, and simplicity.

---

## Algorithms

| Algorithm | Description |
|---|---|
| **Alpha Miner** | Classical algorithm based on footprint matrices and causal relations |
| **Heuristics Miner** | Noise-tolerant miner using dependency measures and frequency filtering |
| **Inductive Miner** | Recursive process tree discovery via sequence/XOR/parallel/loop cuts |

---

## Dataset

**Sepsis Cases – Event Log** (included in `data/`)

A real-world event log from a Dutch hospital recording the treatment process of sepsis patients. Each case is one patient's journey through the hospital, and each event is a clinical activity (e.g., lab test, admission, discharge).

- Format: XES (gzipped)
- Source: [4TU Research Data](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639)

---

## Requirements

- Python 3.10+
- [Graphviz](https://graphviz.org/download/) installed on your system (for visualizations)

Install Python dependencies:

```bash
pip install pm4py==2.7.22.4 pandas numpy graphviz networkx matplotlib scipy
```

---

## Usage

### Run the full analysis (all 3 miners + comparison)

```bash
python main.py
```

This runs the complete pipeline:
1. Loads the Sepsis event log
2. Computes log statistics
3. Discovers models with all three algorithms
4. Evaluates each model (fitness, precision, generalization, simplicity)
5. Prints a comparison table and saves results

### Run a single miner

```bash
# Alpha Miner only (saves JSON summary + PNG visualization)
python src/run_alpha.py

# Heuristics Miner only
python run_sepsis.py

# Inductive Miner only
python src/run_inductive_miner.py
```

---

## Example Output

```
[1/5] Loading event log...
      ✓ Loaded 1050 cases, 15214 events, 16 activities

[2/5] Computing log statistics...
      ✓ Statistics computed

[3/5] Discovering process models...
      ✓ Alpha miner completed
      ✓ Heuristics miner completed
      ✓ Inductive miner completed

[4/5] Evaluating model quality...

[5/5] Comparing models...

     model  fitness  precision  generalization  simplicity
     alpha     0.82       0.41            0.78        0.91
heuristics     0.89       0.53            0.81        0.87
  inductive     0.94       0.61            0.85        0.79
```

---

## Output Files

All results are saved to the `results/` directory:

| File | Written by | Contents |
|---|---|---|
| `results/metrics.csv` | `main.py` | Fitness/precision/generalization/simplicity per miner |
| `results/log_statistics.csv` | `main.py` | Event log statistics (cases, events, activities, durations) |
| `results/alpha_model_summary.json` | `src/run_alpha.py` | Alpha Miner model (places, transitions, arcs) |
| `results/alpha_petri_net.png` | `src/run_alpha.py` | Alpha Miner Petri net visualization |
| `results/heuristics_model_summary.json` | `run_sepsis.py` | Heuristics Miner model (edges, split types, thresholds) |
| `results/inductive_model_summary.json` | `src/run_inductive_miner.py` | Inductive Miner model (cuts, components) |

---

## Project Structure

```
ProcessDiscovery/
├── data/
│   └── Sepsis Cases - Event Log.xes.gz
├── results/                      # Created at runtime
├── src/
│   ├── alpha_miner.py            # Alpha Miner implementation
│   ├── heuristics_miner.py       # Heuristics Miner implementation
│   ├── inductive_miner.py        # Inductive Miner implementation
│   ├── run_alpha.py              # Entry point for Alpha Miner
│   └── run_inductive_miner.py    # Entry point for Inductive Miner
├── main.py                       # Full pipeline (all miners + evaluation)
├── run_sepsis.py                 # Entry point for Heuristics Miner
└── README.md
```

---

## Quality Metrics

| Metric | Measures |
|---|---|
| **Fitness** | How much of the observed behavior the model can reproduce |
| **Precision** | How much the model allows behavior not seen in the log |
| **Generalization** | How well the model generalizes to unseen but valid behavior |
| **Simplicity** | Structural simplicity of the model (arc-degree based) |

All metrics are in the range [0, 1] where higher is better.
