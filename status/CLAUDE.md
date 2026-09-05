# Claude Status

## Completed

- Milestone 1: Topology + Placement Optimizer Foundation.
- Typed GPU topology parsing, configurable relative link costs, single-layer expert
  load summaries and configurable hot-expert detection.
- Linear, round-robin and deterministic greedy placements, fixed assignments,
  memory/count constraints, comparison summaries and explicit communication costs.
- Conservative SLO evaluation/filtering, Pareto frontier and typed recommendations.
- Package installation verified with Python 3.12. 42 tests pass; Ruff and standard
  mypy pass. No Codex-owned modules were implemented or modified.
- Added `moeforge/api/` as the small, stable public surface described in
  CLAUDE_MOEFORGE.md section 8 (`parse_topology`, `calculate_expert_load`,
  `optimize_placement`, `pareto_frontier`). It re-exports the existing tested
  implementations only; no new logic.

## Files Changed

- `moeforge/topology/{__init__,parser,cost_model}.py`
- `moeforge/optimizer/{__init__,load_model,placement,communication_model,slo,pareto,recommendation}.py`
- `moeforge/optimizer/README.md`
- `moeforge/api/{__init__,api}.py` (new: stable public surface, re-exports only)
- `moeforge/shared/__init__.py`, `moeforge/shared/serialization.py`, `INTERFACES.md`
- `tests/conftest.py`, `tests/topology/`, `tests/optimizer/`, `tests/shared/test_serialization.py`,
  `tests/api/test_api.py` (new)
- `.gitignore`, initial package scaffold, `status/CLAUDE.md`
- Shared models, initial contract tests and project metadata were concurrently supplied;
  preserved those versions and integrated against them (shared import formatting only).

## How It Works

- Call `parse_topology`, then `calculate_expert_load` and `optimize_placement` using
  shared dataclasses. Public optimizer functions are exported by `moeforge.optimizer`,
  and also via `moeforge.api` for callers (dashboard, Codex lane) that want the
  minimal stable surface instead of reaching into optimizer/topology internals.
- Greedy scores projected destination GPU load plus weighted incremental communication
  cost. It reserves fixed assignments before placing remaining experts by descending load.
- Supply optional `source_tokens[(source_gpu, expert_id)]` for communication predictions.
  Counts must cover the same expert loads. Without routing origins, costs remain `None`.
- `compare_placements` separates unchanged expert imbalance from projected GPU load.
- `recommend` ties SLO evidence to the supplied benchmark config ID, not the untested plan.
- `filter_slo` and `pareto_frontier` consume synthetic or real `BenchmarkResult` objects.

## How To Test

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e . pytest ruff mypy
.venv/bin/python -m pytest -q
.venv/bin/ruff check moeforge tests
.venv/bin/mypy moeforge
```

Validated with Python 3.12, pytest 9.1.1, Ruff 0.16.6 and mypy 2.3.1.

## Known Issues

- Greedy failure does not prove global infeasibility; no backtracking/local search in M1.
- Only one layer per analysis; callers must include idle experts and avoid overlapping windows.
- Link weights are relative optimizer costs, not bandwidth or latency measurements.
- Unknown link labels require explicit overrides. Affinity columns are tolerated but no
  hardware inventory or NUMA discovery is implemented in this parser milestone.
- Omitted GPU memory budgets mean unconstrained memory; no runtime memory headroom inference.
- Canonical `BenchmarkResult.oom_count` defaults to zero; producers must report OOMs accurately.
- Standard mypy passes. Optional `mypy --strict moeforge` reports one `no-any-return`
  in the concurrently supplied `shared/models.py:to_json_dict` helper; no optimizer exceptions.

## Blocked

- No Milestone 1 blockers. Real telemetry and inference validation await Codex-lane data.

## Interfaces Produced

- Existing shared contracts: `TopologyGraph`, `ExpertLoadSummary`, `PlacementPlan`,
  `SLOEvaluation`; no shared dataclass fields changed.
- Additive local `PlacementComparison` and `Recommendation` dataclasses; optional source
  traffic keyword and serialization semantics documented in `INTERFACES.md`.
- `moeforge.api`: `parse_topology`, `calculate_expert_load`, `optimize_placement`,
  `pareto_frontier` — pure re-exports, not new interfaces.

## Synthetic Data Used

- Eight experts with token counts `[100, 120, 900, 110, 220, 180, 140, 130]`.
- Four GPU topology with two NV4 pairs and SYS links across pairs.
- Explicit synthetic source-to-expert traffic tests demonstrate topology-sensitive choices.
- `BenchmarkResult` fixture uses experiment ID `synthetic-test`; no real benchmark artifacts
  or measured performance claims were generated. Synthetic comparisons are test assertions.

## Next Exact Task

- Await explicit Milestone 2 authorization and Codex-produced `ExpertStats`/`BenchmarkResult`
  artifacts; add integration tests consuming them before real placement comparisons.
