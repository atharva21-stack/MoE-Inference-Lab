"""Shared data contracts between the Codex and Claude implementation lanes.

Defined per INTERFACES.md. Neither lane owns this module exclusively.
Do not restructure these dataclasses without updating INTERFACES.md first.

These models keep MEASURED, PREDICTED, and SYNTHETIC values distinct:
fields named ``predicted_*`` are optimizer-model outputs, not benchmark
measurements. See INTERFACES.md section 28.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

# --- 4. Request-Level Result --------------------------------------------


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


# --- 5. Benchmark Result -------------------------------------------------


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


# --- 6. Parallelism Configuration ----------------------------------------


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    tensor_parallel: int
    data_parallel: int
    expert_parallel: int

    expert_parallel_enabled: bool

    placement_strategy: str
    precision: str


# --- 7. Expert Statistic ---------------------------------------------------


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


# --- 8. Expert Load Summary ------------------------------------------------


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


# --- 9. GPU Node -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GPUNode:
    gpu_id: int

    name: str | None = None
    memory_total_bytes: int | None = None

    numa_node: int | None = None


# --- 10. Topology Link ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopologyLink:
    src_gpu: int
    dst_gpu: int

    link_type: str

    relative_cost: float

    measured_bandwidth_gbps: float | None = None
    measured_latency_us: float | None = None


# --- 11. Topology Graph ------------------------------------------------------


@dataclass(slots=True)
class TopologyGraph:
    gpus: dict[int, GPUNode]
    links: dict[tuple[int, int], TopologyLink]

    def get_link(self, src_gpu: int, dst_gpu: int) -> TopologyLink:
        try:
            return self.links[(src_gpu, dst_gpu)]
        except KeyError:
            raise KeyError(
                f"No topology link recorded between GPU{src_gpu} and GPU{dst_gpu}"
            ) from None

    def cost(self, src_gpu: int, dst_gpu: int) -> float:
        if src_gpu == dst_gpu:
            return 0.0
        return self.get_link(src_gpu, dst_gpu).relative_cost


# --- 12. Placement Constraints ------------------------------------------------


@dataclass(slots=True)
class PlacementConstraints:
    gpu_ids: list[int]

    max_experts_per_gpu: int | None = None

    gpu_memory_budget_bytes: dict[int, int] = field(default_factory=dict)

    expert_memory_bytes: dict[int, int] = field(default_factory=dict)

    fixed_assignments: dict[int, int] = field(default_factory=dict)


# --- 13. Placement Plan ------------------------------------------------------


@dataclass(slots=True)
class PlacementPlan:
    expert_to_gpu: dict[int, int]

    predicted_imbalance: float | None
    predicted_communication_cost: float | None

    feasible: bool

    explanation: list[str] = field(default_factory=list)


# --- 14. Optional Expert Traffic Matrix ---------------------------------------


@dataclass(slots=True)
class ExpertTraffic:
    source_gpu: int
    destination_gpu: int

    token_count: int
    bytes_transferred: int | None = None


# --- 15. SLO ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SLO:
    p95_ttft_ms: float
    p95_tpot_ms: float

    max_error_rate: float = 0.0
    require_zero_oom: bool = True


# --- 16. SLO Evaluation ---------------------------------------------------------


@dataclass(slots=True)
class SLOEvaluation:
    feasible: bool

    ttft_pass: bool | None
    tpot_pass: bool | None
    error_rate_pass: bool
    oom_pass: bool | None

    reasons: list[str] = field(default_factory=list)


# --- 17. Configuration Candidate ------------------------------------------------


@dataclass(slots=True)
class ConfigCandidate:
    config_id: str
    parallel: ParallelConfig
    benchmark: BenchmarkResult | None = None


# --- 18. Pareto Point ------------------------------------------------------------


@dataclass(slots=True)
class ParetoPoint:
    config_id: str

    ttft_p95_ms: float
    tpot_p95_ms: float
    output_tokens_per_second_per_gpu: float


# --- 19. Generation Request --------------------------------------------------------


@dataclass(slots=True)
class GenerationRequest:
    request_id: str
    prompt: str

    max_tokens: int

    temperature: float = 0.0
    top_p: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)


# --- 20. Generation Response --------------------------------------------------------


@dataclass(slots=True)
class GenerationResponse:
    request_id: str
    text: str

    prompt_tokens: int
    output_tokens: int

    token_timestamps_ns: list[int]

    worker_id: str | None = None


# --- 21. Inference Backend Protocol ----------------------------------------------------


class InferenceBackend(Protocol):
    async def health(self) -> bool: ...

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...


# --- JSON serialization helpers (section 3: "JSON-serializable through helper functions") -----


def to_json_dict(value: Any) -> dict[str, Any]:
    """Convert any shared dataclass instance into a plain JSON-serializable dict.

    Dict keys that are tuples (e.g. TopologyGraph.links) are stringified
    since JSON object keys must be strings.
    """
    return _stringify_keys(asdict(value))


def _stringify_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            (str(k) if not isinstance(k, str) else k): _stringify_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj
