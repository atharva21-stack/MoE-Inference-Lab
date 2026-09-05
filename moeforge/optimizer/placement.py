"""Deterministic single-layer placement with explicit capacity constraints."""

import math
from dataclasses import dataclass
from typing import Literal

from moeforge.optimizer.communication_model import communication_cost
from moeforge.optimizer.load_model import calculate_expert_load
from moeforge.shared.models import (
    ExpertStats,
    ExpertTraffic,
    PlacementConstraints,
    PlacementPlan,
    TopologyGraph,
)

# (source GPU, expert ID) -> routed tokens. Optional; never inferred from ExpertStats.gpu_id.
ExpertSources = dict[tuple[int, int], int]


def _validate_constraints(experts: set[int], constraints: PlacementConstraints) -> None:
    ids = constraints.gpu_ids
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("GPU IDs must be unique, nonnegative and nonempty")
    if (
        constraints.max_experts_per_gpu is not None
        and constraints.max_experts_per_gpu < 0
    ):
        raise ValueError("Expert count limit must be nonnegative")
    for mapping, allowed in [
        (constraints.gpu_memory_budget_bytes, set(ids)),
        (constraints.expert_memory_bytes, experts),
    ]:
        if not set(mapping) <= allowed or any(v < 0 for v in mapping.values()):
            raise ValueError("Memory metadata has unknown IDs or negative byte counts")
    if (
        constraints.gpu_memory_budget_bytes
        and set(constraints.expert_memory_bytes) != experts
    ):
        raise ValueError("Memory-constrained placement requires sizes for every expert")
    if not set(constraints.fixed_assignments) <= experts or not set(
        constraints.fixed_assignments.values()
    ) <= set(ids):
        raise ValueError("Fixed assignment references an unknown expert or GPU")


def validate_placement(
    placement: dict[int, int],
    expert_ids: set[int],
    constraints: PlacementConstraints,
) -> list[str]:
    """Return violations; malformed constraints raise ValueError."""
    _validate_constraints(expert_ids, constraints)
    errors = []
    if set(placement) != expert_ids:
        errors.append(
            "Each expert must be placed exactly once; no extra experts are allowed"
        )
    if not set(placement.values()) <= set(constraints.gpu_ids):
        errors.append("Placement references an unknown GPU")
    for e, g in constraints.fixed_assignments.items():
        if placement.get(e) != g:
            errors.append(f"Expert {e} must remain on GPU {g}")
    for g in constraints.gpu_ids:
        assigned = [e for e, gpu in placement.items() if gpu == g]
        limit = constraints.max_experts_per_gpu
        if limit is not None and len(assigned) > limit:
            errors.append(f"GPU {g} exceeds its expert count limit")
        memory = sum(constraints.expert_memory_bytes.get(e, 0) for e in assigned)
        budget = constraints.gpu_memory_budget_bytes.get(g)
        if budget is not None and memory > budget:
            errors.append(f"GPU {g} exceeds its memory budget")
    return errors


def _fits(e: int, g: int, placement: dict[int, int], c: PlacementConstraints) -> bool:
    assigned = [other for other, gpu in placement.items() if gpu == g]
    if c.max_experts_per_gpu is not None and len(assigned) >= c.max_experts_per_gpu:
        return False
    memory = sum(c.expert_memory_bytes.get(other, 0) for other in assigned)
    return g not in c.gpu_memory_budget_bytes or (
        memory + c.expert_memory_bytes.get(e, 0) <= c.gpu_memory_budget_bytes[g]
    )


def _traffic_cost(
    placement: dict[int, int],
    sources: ExpertSources,
    topology: TopologyGraph,
) -> float:
    return communication_cost(
        [
            ExpertTraffic(src, placement[e], tokens)
            for (src, e), tokens in sources.items()
            if e in placement
        ],
        topology,
    )


def _validate_sources(
    sources: ExpertSources | None,
    loads: dict[int, int],
    topology: TopologyGraph,
) -> None:
    if sources is None:
        return
    totals = dict.fromkeys(loads, 0)
    for (src, e), tokens in sources.items():
        if src not in topology.gpus or e not in loads or tokens < 0:
            raise ValueError("Source traffic contains unknown IDs or negative counts")
        totals[e] += tokens
    if totals != loads:
        raise ValueError("Source traffic must account for all routed tokens per expert")


def _plan(
    placement: dict[int, int],
    loads: dict[int, int],
    topology: TopologyGraph,
    constraints: PlacementConstraints,
    sources: ExpertSources | None,
    explanation: list[str],
) -> PlacementPlan:
    errors = validate_placement(placement, set(loads), constraints)
    if errors:
        return PlacementPlan(placement, None, None, False, explanation + errors)
    gpu_loads = [
        sum(loads[e] for e, gpu in placement.items() if gpu == g)
        for g in constraints.gpu_ids
    ]
    mean = sum(gpu_loads) / len(gpu_loads)
    cost = _traffic_cost(placement, sources, topology) if sources is not None else None
    if sources is None:
        explanation.append(
            "Communication cost unavailable: source-to-expert traffic was not supplied."
        )
    explanation.append("Predicted imbalance is peak GPU load divided by mean GPU load.")
    return PlacementPlan(
        placement, max(gpu_loads) / mean if mean else 0.0, cost, True, explanation
    )


def optimize_placement(
    stats: list[ExpertStats],
    topology: TopologyGraph,
    constraints: PlacementConstraints,
    *,
    source_tokens: ExpertSources | None = None,
    topology_weight: float = 1.0,
) -> PlacementPlan:
    """Greedy projected GPU load + weighted incremental communication cost.

    Ties use GPU ID; expert ties use expert ID. No randomness is needed.
    Failure means greedy could not complete, not proof no feasible solution exists.
    """
    loads = calculate_expert_load(stats).tokens_by_expert
    _validate_constraints(set(loads), constraints)
    if not set(constraints.gpu_ids) <= set(topology.gpus):
        raise ValueError("Constraint GPU is absent from topology")
    if not math.isfinite(topology_weight) or topology_weight < 0:
        raise ValueError("Topology weight must be finite and nonnegative")
    _validate_sources(source_tokens, loads, topology)
    placement: dict[int, int] = {}
    for e, g in sorted(constraints.fixed_assignments.items()):
        if not _fits(e, g, placement, constraints):
            return PlacementPlan(
                {}, None, None, False, ["Fixed assignments exceed GPU capacity."]
            )
        placement[e] = g
    explanation = [
        "Greedy placement minimizes projected GPU token load plus weighted relative cost."
    ]
    for e in sorted(loads, key=lambda e: (-loads[e], e)):
        if e in placement:
            continue
        candidates = [
            g
            for g in sorted(constraints.gpu_ids)
            if _fits(e, g, placement, constraints)
        ]
        if not candidates:
            return PlacementPlan(
                placement,
                None,
                None,
                False,
                explanation
                + [
                    f"No remaining GPU capacity for expert {e}; greedy does not prove infeasibility.",
                ],
            )

        def score(g: int, expert: int = e) -> float:
            projected = (
                sum(loads[other] for other, gpu in placement.items() if gpu == g)
                + loads[expert]
            )
            penalty = (
                _traffic_cost({expert: g}, source_tokens, topology)
                if source_tokens is not None
                else 0
            )
            return projected + topology_weight * penalty

        g = min(candidates, key=lambda g: (score(g), g))
        explanation.append(
            f"Expert {e} ({loads[e]} routed tokens) assigned to GPU {g}; score {score(g):g}."
        )
        placement[e] = g
    return _plan(placement, loads, topology, constraints, source_tokens, explanation)


def baseline_placement(
    stats: list[ExpertStats],
    topology: TopologyGraph,
    constraints: PlacementConstraints,
    strategy: Literal["linear", "round_robin"],
    *,
    source_tokens: ExpertSources | None = None,
) -> PlacementPlan:
    loads = calculate_expert_load(stats).tokens_by_expert
    _validate_constraints(set(loads), constraints)
    _validate_sources(source_tokens, loads, topology)
    if not set(constraints.gpu_ids) <= set(topology.gpus):
        raise ValueError("Constraint GPU is absent from topology")
    ids = sorted(constraints.gpu_ids)
    experts = sorted(loads)
    if strategy == "round_robin":
        placement = {e: ids[i % len(ids)] for i, e in enumerate(experts)}
    elif strategy == "linear":
        q, r = divmod(len(experts), len(ids))
        slots = [g for i, g in enumerate(ids) for _ in range(q + (i < r))]
        placement = dict(zip(experts, slots, strict=True))
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")
    return _plan(
        placement,
        loads,
        topology,
        constraints,
        source_tokens,
        [f"{strategy} baseline."],
    )


@dataclass(slots=True)
class PlacementComparison:
    strategy: str
    plan: PlacementPlan
    max_gpu_load: int | None
    mean_gpu_load: float | None
    expert_imbalance: float
    communication_cost: float | None
    memory_feasible: bool


def compare_placements(
    stats: list[ExpertStats],
    topology: TopologyGraph,
    constraints: PlacementConstraints,
    *,
    source_tokens: ExpertSources | None = None,
    topology_weight: float = 1.0,
) -> list[PlacementComparison]:
    summary = calculate_expert_load(stats)
    plans = [
        (
            s,
            baseline_placement(
                stats, topology, constraints, s, source_tokens=source_tokens
            ),
        )
        for s in ("linear", "round_robin")
    ]
    plans.append(
        (
            "moeforge",
            optimize_placement(
                stats,
                topology,
                constraints,
                source_tokens=source_tokens,
                topology_weight=topology_weight,
            ),
        )
    )
    comparisons = []
    for strategy, plan in plans:
        gpu_loads = [
            sum(
                summary.tokens_by_expert[e]
                for e, gpu in plan.expert_to_gpu.items()
                if gpu == g
            )
            for g in constraints.gpu_ids
        ]
        errors = validate_placement(
            plan.expert_to_gpu, set(summary.tokens_by_expert), constraints
        )
        memory_ok = set(plan.expert_to_gpu) == set(
            summary.tokens_by_expert
        ) and not any("memory budget" in error for error in errors)
        comparisons.append(
            PlacementComparison(
                strategy,
                plan,
                max(gpu_loads) if plan.feasible else None,
                sum(gpu_loads) / len(gpu_loads) if plan.feasible else None,
                summary.imbalance_ratio,
                plan.predicted_communication_cost,
                memory_ok,
            )
        )
    return comparisons
