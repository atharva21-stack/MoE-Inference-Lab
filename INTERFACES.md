# MoEForge — Shared Interfaces

> This file defines the shared contracts between the Codex and Claude implementation lanes.
> Both agents must treat these interfaces as stable unless a change is explicitly documented here.

---

# 1. Purpose

MoEForge is being developed in parallel by two implementation lanes:

- **Codex lane:** serving, benchmarking, telemetry, storage, CLI, kernels
- **Claude lane:** topology, placement optimization, SLO analysis, Pareto analysis, dashboard/API

The lanes must communicate through small, typed contracts.

Do not import implementation details across ownership boundaries when a shared interface is sufficient.

---

# 2. Shared Python Location

Create the shared models in:

```text
moeforge/shared/models.py
```

Optional supporting enums may live in:

```text
moeforge/shared/enums.py
```

Both lanes may import from `moeforge.shared`.

Neither lane owns `moeforge/shared/` exclusively.

Changes to shared models require:

1. updating this document,
2. updating affected tests,
3. documenting the change in the lane's status file.

---

# 3. Common Rules

All shared models should:

- use Python 3.12+ type hints,
- use `dataclasses.dataclass` unless Pydantic is clearly needed,
- avoid runtime-specific objects,
- be JSON-serializable through helper functions,
- use milliseconds for latency fields unless stated otherwise,
- use bytes for memory fields,
- use integer GPU IDs beginning at 0,
- distinguish **measured** values from **predicted** values,
- never use fabricated defaults for performance measurements.

Use `None` when a measurement is unavailable.

---

# 4. Request-Level Result

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(slots=True)
class RequestResult:
    request_id: str

    prompt_tokens: int
    output_tokens: int

    queue_ms: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float

    status: Literal["success", "error"]
    error_message: str | None = None

    worker_id: str | None = None
```

## Semantics

### `queue_ms`

Time spent waiting before inference execution begins, when available.

### `ttft_ms`

Time from request submission to first generated token.

### `tpot_ms`

Mean time per output token **after the first generated token**.

Recommended calculation:

```text
tpot_ms =
(last_token_timestamp - first_token_timestamp)
/
max(output_tokens - 1, 1)
```

If fewer than 2 output tokens are generated, `tpot_ms` may be `None`.

### `e2e_ms`

Total request wall-clock latency.

---

# 5. Benchmark Result

Codex produces this object.

Claude consumes it for SLO analysis and Pareto selection.

```python
from dataclasses import dataclass

@dataclass(slots=True)
class BenchmarkResult:
    experiment_id: str
    config_id: str

    gpu_count: int
    request_count: int
    successful_requests: int
    failed_requests: int

    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    ttft_p99_ms: float | None

    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    tpot_p99_ms: float | None

    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    e2e_p99_ms: float | None

    input_tokens_per_second: float | None
    output_tokens_per_second: float | None
    total_tokens_per_second: float | None
    requests_per_second: float | None

    output_tokens_per_second_per_gpu: float | None

    error_rate: float

    oom_count: int = 0
```

## Required Derived Field

When possible:

```text
output_tokens_per_second_per_gpu =
output_tokens_per_second / gpu_count
```

Do not calculate it when `gpu_count <= 0` or throughput is unavailable.

---

# 6. Parallelism Configuration

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ParallelConfig:
    tensor_parallel: int
    data_parallel: int
    expert_parallel: int

    expert_parallel_enabled: bool

    placement_strategy: str
    precision: str
```

Examples:

```python
ParallelConfig(
    tensor_parallel=1,
    data_parallel=4,
    expert_parallel=4,
    expert_parallel_enabled=True,
    placement_strategy="linear",
    precision="bf16",
)
```

Valid placement strategy strings initially:

```text
linear
round_robin
eplb
moeforge
```

Do not assume every runtime supports every configuration.

Runtime-specific validation belongs outside this shared model.

---

# 7. Expert Statistic

Codex produces this object from router telemetry.

Claude consumes it.

```python
from dataclasses import dataclass

@dataclass(slots=True)
class ExpertStats:
    layer_id: int
    expert_id: int
    gpu_id: int

    routed_tokens: int

    window_start_ns: int | None = None
    window_end_ns: int | None = None

    mean_expert_load: float | None = None
    imbalance_ratio: float | None = None
```

## Rules

`routed_tokens` must represent observed tokens routed to the expert during the aggregation window.

`mean_expert_load` and `imbalance_ratio` may be left `None` by the telemetry producer because the optimizer can recompute them.

The optimizer must never trust a derived metric blindly when it can calculate the value from raw token counts.

---

# 8. Expert Load Summary

Claude produces this internally from `ExpertStats`.

```python
from dataclasses import dataclass

@dataclass(slots=True)
class ExpertLoadSummary:
    total_routed_tokens: int
    mean_expert_load: float
    max_expert_load: int

    imbalance_ratio: float
    balancedness: float

    tokens_by_expert: dict[int, int]
    tokens_by_gpu: dict[int, int]

    hot_experts: list[int]
```

## Formulas

```text
mean_expert_load =
total_routed_tokens / number_of_experts
```

```text
imbalance_ratio =
max_expert_load / mean_expert_load
```

```text
balancedness =
mean_expert_load / max_expert_load
```

When no routed tokens exist:

```text
imbalance_ratio = 0.0
balancedness = 1.0
```

---

# 9. GPU Node

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class GPUNode:
    gpu_id: int

    name: str | None = None
    memory_total_bytes: int | None = None

    numa_node: int | None = None
```

---

# 10. Topology Link

Claude produces this object.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TopologyLink:
    src_gpu: int
    dst_gpu: int

    link_type: str

    relative_cost: float

    measured_bandwidth_gbps: float | None = None
    measured_latency_us: float | None = None
```

## Important

`relative_cost` is an optimizer weight.

It is **not** a bandwidth or latency claim.

Measured values must use their dedicated fields.

---

# 11. Topology Graph

```python
from dataclasses import dataclass

@dataclass(slots=True)
class TopologyGraph:
    gpus: dict[int, GPUNode]
    links: dict[tuple[int, int], TopologyLink]
```

Required convenience operations may include:

```python
def get_link(self, src_gpu: int, dst_gpu: int) -> TopologyLink:
    ...
```

```python
def cost(self, src_gpu: int, dst_gpu: int) -> float:
    ...
```

Do not require NetworkX in the public interface.

The implementation may use NetworkX internally.

---

# 12. Placement Constraints

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class PlacementConstraints:
    gpu_ids: list[int]

    max_experts_per_gpu: int | None = None

    gpu_memory_budget_bytes: dict[int, int] = field(default_factory=dict)

    expert_memory_bytes: dict[int, int] = field(default_factory=dict)

    fixed_assignments: dict[int, int] = field(default_factory=dict)
```

`fixed_assignments` maps:

```text
expert_id -> gpu_id
```

Use it for experts that may not move.

---

# 13. Placement Plan

Claude produces this object.

Codex may later consume it when applying an optimized placement.

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class PlacementPlan:
    expert_to_gpu: dict[int, int]

    predicted_imbalance: float | None
    predicted_communication_cost: float | None

    feasible: bool

    explanation: list[str] = field(default_factory=list)
```

## Critical Rule

Fields beginning with `predicted_` are model predictions.

They must never be presented as measured benchmark improvements.

The real improvement is determined only after Codex runs a benchmark and emits a new `BenchmarkResult`.

---

# 14. Optional Expert Traffic Matrix

For topology-aware communication analysis:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class ExpertTraffic:
    source_gpu: int
    destination_gpu: int

    token_count: int
    bytes_transferred: int | None = None
```

This is optional for Milestone 1.

A later telemetry implementation may provide it.

---

# 15. SLO

Claude owns SLO evaluation.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SLO:
    p95_ttft_ms: float
    p95_tpot_ms: float

    max_error_rate: float = 0.0
    require_zero_oom: bool = True
```

---

# 16. SLO Evaluation

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class SLOEvaluation:
    feasible: bool

    ttft_pass: bool | None
    tpot_pass: bool | None
    error_rate_pass: bool
    oom_pass: bool | None

    reasons: list[str] = field(default_factory=list)
```

When a required metric is unavailable, the evaluator should not silently mark it as passing.

It should return:

```text
feasible = False
```

and explain that the required measurement is missing.

---

# 17. Configuration Candidate

Later configuration search may use:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class ConfigCandidate:
    config_id: str
    parallel: ParallelConfig
    benchmark: BenchmarkResult | None = None
```

Do not conflate a candidate with a measured benchmark.

---

# 18. Pareto Point

Claude may use:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class ParetoPoint:
    config_id: str

    ttft_p95_ms: float
    tpot_p95_ms: float
    output_tokens_per_second_per_gpu: float
```

A configuration A dominates B when A is:

```text
no worse in all objectives
AND
strictly better in at least one objective
```

where:

```text
TTFT: lower is better
TPOT: lower is better
tokens/sec/GPU: higher is better
```

---

# 19. Generation Request

Codex owns runtime implementation but uses this shared request shape.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class GenerationRequest:
    request_id: str
    prompt: str

    max_tokens: int

    temperature: float = 0.0
    top_p: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)
```

---

# 20. Generation Response

```python
from dataclasses import dataclass

@dataclass(slots=True)
class GenerationResponse:
    request_id: str
    text: str

    prompt_tokens: int
    output_tokens: int

    token_timestamps_ns: list[int]

    worker_id: str | None = None
```

The token timestamp list allows the benchmark layer to calculate TTFT and TPOT without runtime-specific assumptions.

For a mock backend, synthetic timestamps are acceptable only in tests and must be explicitly test-generated.

---

# 21. Inference Backend Protocol

Codex owns implementations.

Shared contract:

```python
from typing import Protocol

class InferenceBackend(Protocol):

    async def health(self) -> bool:
        ...

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        ...
```

Initial implementations:

```text
MockBackend
VLLMBackend
```

Future:

```text
DynamoBackend
```

---

# 22. Result Artifact Contract

Every benchmark run should eventually produce:

```text
results/<experiment-id>/
├── config.yaml
├── environment/
│   ├── hardware.json
│   ├── software.json
│   └── topology.txt
├── requests.parquet
├── experts.parquet          # once expert telemetry exists
├── gpu.parquet              # once GPU telemetry exists
├── summary.json
└── report.md
```

`summary.json` should serialize the `BenchmarkResult` plus experiment metadata.

Do not manually insert performance values into `summary.json`.

---

# 23. Integration Boundary

## Codex -> Claude

Codex provides:

```text
BenchmarkResult
ExpertStats
optional ExpertTraffic
raw result artifacts
```

Claude should not need to import:

```text
vLLM backend internals
HTTP client code
stream parser code
Prometheus collector internals
```

---

## Claude -> Codex

Claude provides:

```text
TopologyGraph
PlacementPlan
SLOEvaluation
Pareto results
```

Codex should not need to import:

```text
greedy optimizer internals
topology parser internals
Pareto implementation details
```

---

# 24. Milestone 1 Compatibility

During the first milestone:

## Codex

Real outputs required:

```text
RequestResult
BenchmarkResult
```

Synthetic or stub output acceptable:

```text
ExpertStats
```

because real expert telemetry is Milestone 2.

---

## Claude

Real implementation required:

```text
TopologyGraph
ExpertLoadSummary
PlacementPlan
SLOEvaluation
Pareto calculation
```

Synthetic inputs acceptable:

```text
ExpertStats
BenchmarkResult
```

until Codex produces real artifacts.

---

# 25. Change Policy

If an agent believes a shared interface must change:

1. Do not silently edit only the Python implementation.
2. Update this file.
3. Explain:
   - what changed,
   - why,
   - backward-compatibility impact.
4. Add or update shared contract tests.
5. Note the change in:
   - `status/CODEX.md`, or
   - `status/CLAUDE.md`.

Prefer additive changes over breaking changes.

---

# 26. Shared Contract Tests

Create:

```text
tests/shared/test_models.py
```

At minimum test:

```text
BenchmarkResult construction
ExpertStats construction
TopologyGraph construction
PlacementPlan construction
JSON serialization where implemented
```

Both lanes must keep these tests passing.

---

# 27. Canonical Milestone 1 Data Flow

```text
Mock/VLLM Backend
       │
       ▼
Benchmark Client
       │
       ▼
RequestResult[]
       │
       ▼
BenchmarkResult
       │
       ├───────────────► SLO Evaluation
       │
       └───────────────► Pareto Analysis


Synthetic ExpertStats
       │
       ▼
Expert Load Summary
       │
       ▼
TopologyGraph
       │
       ▼
Placement Optimizer
       │
       ▼
PlacementPlan
```

Milestone 2 replaces synthetic `ExpertStats` with real router telemetry.

---

# 28. One Principle Above Everything Else

MoEForge must keep these concepts separate:

```text
MEASURED
PREDICTED
SYNTHETIC
```

A value from a real benchmark is measured.

A value from the optimizer's cost model is predicted.

A value in a unit-test fixture is synthetic.

Never blur those categories in code, reports, dashboards, README claims, or resume bullets.

## Milestone 1 implementation notes (Claude lane)

- The canonical dataclasses above are implemented unchanged in `moeforge/shared/models.py`
  and re-exported from `moeforge.shared`. This introduces no schema changes.
- `moeforge.shared.serialization.to_json` encodes dataclasses recursively. JSON object
  keys are strings; `TopologyGraph.links` uses stringified tuple keys through the shared
  `to_json_dict` helper because JSON cannot represent tuple dictionary keys. This is
  an additive encoding helper, not a change to the Python graph contract.
- Placement and load analysis accept one layer per call. Include zero-token experts
  and non-overlapping aggregation windows. Mixed layers are rejected to avoid merging
  unrelated experts with the same integer ID.
- `PlacementPlan.predicted_imbalance` is the projected **GPU** peak-to-mean load ratio.
  `ExpertLoadSummary.imbalance_ratio` remains the **expert** peak-to-mean ratio; moving
  experts does not change their routed token counts.
- Optional optimizer keyword `source_tokens` maps `(source_gpu, expert_id)` to token
  counts for the same layer/window. Counts must cover all expert loads. This supplies
  the routing origin absent from `ExpertStats`; without it, communication predictions
  remain `None` and greedy uses load alone. This does not alter `ExpertTraffic`.
- `Recommendation` (in `moeforge.optimizer.recommendation`) wraps a `PlacementPlan`,
  optional `benchmark_slo`, `benchmark_config_id`, and an explanation. SLO evidence
  applies only to the supplied benchmark, never automatically to a new placement.
- Pareto analysis excludes missing/nonfinite/negative objectives and nonpositive GPU
  counts, preserves input order, and retains equal non-dominated points. SLO filtering
  is a separate explicit operation.
