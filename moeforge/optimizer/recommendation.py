"""Recommendations preserve the boundary between predictions and benchmark evidence."""

from dataclasses import dataclass

from moeforge.optimizer.placement import ExpertSources, optimize_placement
from moeforge.optimizer.slo import evaluate_slo
from moeforge.shared.models import (
    SLO,
    BenchmarkResult,
    ExpertStats,
    PlacementConstraints,
    PlacementPlan,
    SLOEvaluation,
    TopologyGraph,
)


@dataclass(slots=True)
class Recommendation:
    placement: PlacementPlan
    benchmark_slo: SLOEvaluation | None
    benchmark_config_id: str | None
    explanation: list[str]


def recommend(
    stats: list[ExpertStats],
    topology: TopologyGraph,
    constraints: PlacementConstraints,
    *,
    benchmark: BenchmarkResult | None = None,
    slo: SLO | None = None,
    source_tokens: ExpertSources | None = None,
    topology_weight: float = 1.0,
) -> Recommendation:
    if (benchmark is None) != (slo is None):
        raise ValueError("Supply both benchmark and SLO, or neither")
    plan = optimize_placement(
        stats,
        topology,
        constraints,
        source_tokens=source_tokens,
        topology_weight=topology_weight,
    )
    evaluation = (
        evaluate_slo(benchmark, slo)
        if benchmark is not None and slo is not None
        else None
    )
    return Recommendation(
        plan,
        evaluation,
        benchmark.config_id if benchmark else None,
        plan.explanation
        + [
            "Placement values are predictions; validate with a new benchmark.",
            "SLO evaluation applies only to the supplied benchmark configuration.",
        ],
    )
