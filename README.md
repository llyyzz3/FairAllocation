# Fair Allocation of Indivisible Goods

This repository contains the implementation and experimental framework for the MSc dissertation **Fair Allocation of Indivisible Goods: Experiments and Implementation** (University of Edinburgh, 2026).

The project studies how fair-allocation algorithms for indivisible goods behave in practice. It implements classical baselines, approximate EFX and EFkX algorithms, and a specialised exact EF2X algorithm for four agents. The experiments evaluate fairness guarantees, empirical approximation quality, and runtime on synthetic valuation profiles and Spliddit data.

## Algorithms

| Implementation | Class | Intended guarantee / role |
| --- | --- | --- |
| Round-Robin | `algorithms.baselines.RoundRobin` | Simple deterministic baseline |
| Envy-Cycle Elimination | `algorithms.baselines.EnvyCycleElimination` | EF1 under non-negative additive valuations |
| Draft-and-Eliminate | `algorithms.baselines.EFX_0618_Approx` | Approximately `0.618`-EFX and EF1 |
| k-Round-Robin with ECE | `algorithms.kgoods.KRoundRobinWithECE` | `k/(k+1)`-EFkX |
| Approximate EFkX | `algorithms.EFKX.ApproximateEFkX` | `(k+1)/(k+2)`-EFkX for `k >= 2` |
| Four-agent EF2X | `algorithms.ef2x_four_agents.EF2XFourAgents` | Exact EF2X for four agents |

All implementations assume non-negative additive valuations. Agents and goods are represented by zero-based integer indices.

## Repository structure

```text
FairAllocation/
|-- algorithms/
|   |-- baselines.py              # Round-Robin, ECE, and Draft-and-Eliminate
|   |-- kgoods.py                 # k-Round-Robin with ECE
|   |-- EFKX.py                   # Approximate EFkX
|   |-- ef2x_four_agents.py       # specialised four-agent EF2X algorithm
|   `-- ef2x_utils.py             # shared EF2X utilities
|-- data/
|   |-- generator.py              # instances, synthetic generators, Spliddit loader
|   |-- spliddit-2026-07-18-goods.sql
|   |-- spliddit_goods_data_full_compiled.csv
|   `-- spliddit_goods_data_full_compiled.xlsx
|-- evaluation/
|   `-- metrics.py                # EF1, EFX, Max-alpha-EFkX, and welfare metrics
|-- results/                      # representative console logs
|-- main.py                       # general benchmark grid
`-- benchmark_4agents.py          # focused four-agent EF2X benchmark
```

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/llyyzz3/FairAllocation.git
cd FairAllocation

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy networkx matplotlib scipy
```

## Core data model

An allocation instance contains:

- `n`: number of agents;
- `m`: number of goods;
- `valuations`: an `n x m` NumPy array, where `valuations[i][g]` is agent `i`'s value for good `g`;
- `dist_type`: a label used in experiment output.

Algorithms expose a common `run(instance)` interface and return:

```python
{
    0: [0, 3],
    1: [1, 5],
    2: [2, 6],
    3: [4, 7],
}
```

Each key is an agent index and each value is the list of goods allocated to that agent.

## Using the modules

### Create an instance and run an algorithm

```python
from data.generator import FixedGenerator
from algorithms.baselines import RoundRobin
from evaluation.metrics import EF1Checker, EFXChecker, MaxAlphaEFkX

valuations = [
    [9, 8, 2, 1, 5, 4, 3, 7],
    [8, 9, 1, 2, 4, 5, 7, 3],
    [2, 1, 9, 8, 3, 7, 5, 4],
    [1, 2, 8, 9, 7, 3, 4, 5],
]

instance = FixedGenerator(
    valuations,
    name="example",
).generate(n=4, m=8)

algorithm = RoundRobin()
allocation = algorithm.run(instance)

print("allocation:", allocation)
print("EF1:", EF1Checker().compute(instance, allocation))
print("EFX:", EFXChecker().compute(instance, allocation))
print("Max alpha-EF2X:", MaxAlphaEFkX(k=2).compute(instance, allocation))
```

Other algorithms use the same `run(instance)` interface:

```python
from algorithms.baselines import EnvyCycleElimination, EFX_0618_Approx
from algorithms.kgoods import KRoundRobinWithECE
from algorithms.EFKX import ApproximateEFkX

algorithms = [
    EnvyCycleElimination(),
    EFX_0618_Approx(),
    KRoundRobinWithECE(k=2),
    ApproximateEFkX(k=2),  # requires k >= 2 and normally m >= n
]

for algorithm in algorithms:
    allocation = algorithm.run(instance)
    print(algorithm.name, allocation)
```

### Use the specialised four-agent EF2X algorithm

The specialised implementation requires exactly four agents. `run(instance)` provides the common benchmark interface, while `solve()` returns diagnostic information such as the stage trace and number of iterations.

```python
from algorithms.ef2x_four_agents import EF2XFourAgents

solver = EF2XFourAgents(instance.valuations)

# Common interface
allocation = solver.run(instance)
print(allocation)

# Detailed result
result = solver.solve()
print(result.allocation)
print(result.iterations)
print([stage.stage.value for stage in result.stage_trace])
```

### Generate synthetic instances

```python
import numpy as np

from data.generator import (
    UniformGenerator,
    NormalGenerator,
    SparseZeroHeavyGenerator,
    SharedTopGoodsGenerator,
    MallowsGenerator,
    IdenticalGenerator,
    OrderedGenerator,
)

np.random.seed(42)

generators = [
    UniformGenerator(v_min=0, v_max=1000),
    NormalGenerator(mean=500, std=100),
    SparseZeroHeavyGenerator(zero_probability=0.8),
    SharedTopGoodsGenerator(shared_top_fraction=0.2),
    MallowsGenerator(phi=0.5),
    IdenticalGenerator(),
    OrderedGenerator(),
]

for generator in generators:
    generated_instance = generator.generate(n=4, m=16)
    print(generator.name, generated_instance.valuations.shape)
```

### Load Spliddit instances

```python
from data.generator import SplidditSQLDataset

dataset = SplidditSQLDataset("data/spliddit-2026-07-18-goods.sql")
instances = dataset.load_instances(
    require_m_at_least_n=False,
    require_strictly_more_items=True,
    require_positive_valuations=True,
    include_demo=False,
)

print(f"Loaded {len(instances)} instances")
```

Check the source dataset's licence and terms before redistributing or reusing it.

## Running the benchmarks

### General benchmark grid

```bash
python main.py
```

`main.py` currently runs:

- agent-good sizes from `(3, 12)` to `(8, 32)`;
- 13 synthetic valuation environments;
- `k` values from 1 to 5;
- five repetitions with deterministic seeds;
- the full applicable algorithm suite.

This is a substantial experiment and can create many result files. For a quick test, reduce `size_grid`, `data_environments`, `k_values`, and `repeats` near the bottom of `main.py`.

You can also call the benchmark functions directly:

```python
from algorithms.baselines import RoundRobin, EnvyCycleElimination
from data.generator import UniformGenerator
from evaluation.metrics import EF1Checker, EFXChecker
from main import run_experiment_suite

run_experiment_suite(
    n=4,
    m=16,
    data_envs=[UniformGenerator(v_min=0, v_max=1000)],
    algos_to_test=[RoundRobin(), EnvyCycleElimination()],
    metrics_to_check=[EF1Checker(), EFXChecker()],
    scenario={"name": "quick_start", "seed": 42, "repeat": 1},
)
```

### Focused four-agent EF2X benchmark

```bash
python benchmark_4agents.py
```

This benchmark runs the exact four-agent EF2X implementation on four fixed regression cases and randomly generated instances with:

- exactly four agents;
- `m` in `{8, 12, 16, 24, 32, 50}`;
- Uniform, Normal, Mallows, Identical, and Ordered profiles;
- deterministic seeds;
- EF1, EFX, Max-alpha-EFX, and internal EF2X validation.

For programmatic control:

```python
from benchmark_4agents import run_4agents_benchmark

run_4agents_benchmark(repeats=1, base_seed=20260714)
```

## Outputs

Benchmark runs create:

```text
results/
|-- console_logs/     # complete terminal output
`-- result_files/     # one JSON report per algorithm-instance run
```

Each JSON report records experiment metadata, the valuation matrix, allocation, fairness metrics, runtime, and failure details when an algorithm raises an exception.

Three representative console logs are committed under [`results/`](results/). Large generated result collections should normally remain untracked.

## Evaluation metrics

The evaluation layer is independent of the allocation algorithms and includes:

- `EF1Checker`: exact Boolean EF1 validation;
- `EFXChecker`: exact Boolean EFX validation using positive-good deletion;
- `EmpiricalEFXRatio`: proportion of ordered agent pairs satisfying EFX;
- `MaxAlphaEFkX(k)`: maximum achieved approximation factor, with `k=1` corresponding to Max-alpha-EFX;
- `NashSocialWelfare`: welfare evaluation.

A Max-alpha value of `1` means that the allocation satisfies the corresponding exact fairness notion.

## Experimental summary

The dissertation reports that all implemented algorithms satisfied their intended guarantees on the tested instances. Empirical fairness was often stronger than the worst-case theoretical guarantee, but performance depended on the valuation profile. Envy-Cycle Elimination generally improved on Round-Robin, while Approximate EFkX produced more stable EFkX results than k-Round-Robin with ECE at the same `k`. The specialised four-agent algorithm achieved exact EF2X throughout the reported tests, at the cost of substantially higher and more instance-dependent runtime than the approximate alternatives.

These results are empirical observations from the tested synthetic and Spliddit instances, not universal performance guarantees.

## Reproducibility notes

- Random experiments use explicit NumPy seeds.
- Tie-breaking is deterministic in the implemented algorithms where applicable.
- The specialised EF2X solver supports exactly four agents.
- `ApproximateEFkX` requires `k >= 2`; its seed phase generally requires at least one good per agent.
- Do not commit `__pycache__/`, `.pyc` files, or the full generated `results/result_files/` directory.

## Academic context

This repository accompanies:

> Yuzhuo Li. *Fair Allocation of Indivisible Goods: Experiments and Implementation*. MSc dissertation, School of Informatics, University of Edinburgh, 2026.

If you use this repository in academic work, please cite the dissertation and the original algorithm papers discussed there.


