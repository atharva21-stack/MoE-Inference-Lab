# MoEForge

A Python starter template for topology-aware Mixture-of-Experts inference analysis.
Bring your own model, serving backend, traffic source, hardware, and measurements.

The repository includes reusable topology and placement analysis. It does not bundle
workload profiles, prompt datasets, model-specific settings, or benchmark results.
Small synthetic fixtures exist only to test correctness without GPUs.

## Get started

Requires Python 3.12 or newer. No GPU is needed to install or run the tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

## What's included

| Component | Purpose |
| --- | --- |
| `moeforge/shared/` | Typed data contracts for runtime results and analysis |
| `moeforge/topology/` | Parse GPU topology and configure relative link costs |
| `moeforge/optimizer/` | Load analysis, placement constraints, greedy placement, SLO and Pareto analysis |
| `moeforge/api/` | Python imports for the core analysis functions |
| `tests/` | Synthetic correctness tests |

`moeforge.api` is a Python API, not an HTTP server. Serving, telemetry collection,
benchmark execution, and deployment are extension points, not implemented features.

## Use your own data

Start with `parse_topology(text)` using your topology matrix. Supply a list of
`ExpertStats` for one layer and a `PlacementConstraints` object describing your GPU
IDs, expert sizes, capacity limits, and fixed assignments. Pass these to
`optimize_placement(stats, topology, constraints)`.

Use `evaluate_slo(result, slo)` to check your `BenchmarkResult` and
`pareto_frontier(candidates)` to compare measured candidates. See
[the data contracts](INTERFACES.md) and [optimizer usage](moeforge/optimizer/README.md).

An optimizer prediction is not a measured improvement. Validate a proposed placement
with your own runtime and reproducible benchmark. Missing communication-source data
produces an unavailable communication prediction, not an assumed zero cost.

## Adapt this template

1. Create your own repository from this project or clone it.
2. Set the project name and metadata in `pyproject.toml`.
3. Connect your data producer using [the extension guide](docs/EXTENDING.md).
4. Supply your own configuration and workload outside the reusable analysis modules.
5. Keep measurements separate from predictions and synthetic test fixtures.

The package name remains `moeforge`; rename its directory, imports, and package-discovery
pattern together if you change it. Choose a license before redistributing your own work.

## Development

```bash
python -m pytest
ruff check moeforge tests
mypy moeforge
```

GitHub Actions runs these checks on pushes and pull requests. Local agent instructions
and handoff files are excluded from the template.
