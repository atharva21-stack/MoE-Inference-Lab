# Topology and placement foundation

This module analyzes one MoE layer at a time using the contracts in `INTERFACES.md`.
It does not run inference or apply placements. Tests use synthetic data exclusively.

```python
from moeforge.optimizer import compare_placements, recommend
from moeforge.shared import ExpertStats, PlacementConstraints
from moeforge.topology import parse_topology

graph = parse_topology("GPU0 GPU1\nGPU0 X NV4\nGPU1 NV4 X")
synthetic_stats = [ExpertStats(0, 0, 0, 100), ExpertStats(0, 1, 0, 900)]
constraints = PlacementConstraints(gpu_ids=[0, 1], max_experts_per_gpu=1)
comparison = compare_placements(synthetic_stats, graph, constraints)
recommendation = recommend(synthetic_stats, graph, constraints)
```

Include all experts, including idle ones, and supply non-overlapping windows. Hot
experts exceed mean load plus a configurable token threshold (default zero).
Expert imbalance does not change when experts move; projected GPU imbalance can.

Greedy processes fixed assignments first, then descending expert load, breaking
ties by expert and GPU ID. Each choice minimizes projected destination GPU load
plus `topology_weight * incremental_communication_cost`. It respects count and
explicit memory limits. A memory budget requires sizes for all experts; omitted
GPU budgets mean unconstrained memory. No automatic HBM/headroom inference occurs.
A greedy failure is not proof that every possible assignment is infeasible.
Baselines remain literal contiguous/round-robin mappings and report constraint
violations rather than silently turning into a different strategy.

Communication requires optional `(source_gpu, expert_id) -> tokens` counts covering
the same expert loads. Expert residence is not routing origin. Without source data,
communication cost is unavailable. With it, costs are token-weighted relative
optimizer weights, not time predictions. Link cost overrides support new labels;
unknown labels otherwise fail explicitly. Dedicated measured link fields remain
available for a future time-based model; V1 never mixes their units with weights.

SLO checks reject missing or invalid latency measurements. The canonical benchmark
contract defaults `oom_count` to zero: producers must populate it accurately when
OOM checking is required. Pareto analysis excludes incomplete or invalid objectives;
apply `filter_slo` first if only SLO-feasible candidates should be considered.

Run validation from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e . pytest ruff mypy
.venv/bin/python -m pytest
.venv/bin/ruff check moeforge tests
.venv/bin/mypy moeforge
```
