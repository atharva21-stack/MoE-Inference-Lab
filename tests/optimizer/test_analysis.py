from dataclasses import replace

import pytest

from moeforge.optimizer.pareto import pareto_frontier
from moeforge.optimizer.recommendation import recommend
from moeforge.optimizer.slo import evaluate_slo, filter_slo
from moeforge.shared.models import (
    SLO,
    BenchmarkResult,
    ExpertStats,
    PlacementConstraints,
    TopologyGraph,
)


def test_slo(benchmark: BenchmarkResult) -> None:
    slo = SLO(10, 2)
    assert evaluate_slo(benchmark, slo).feasible
    for result in [
        replace(benchmark, ttft_p95_ms=None),
        replace(benchmark, tpot_p95_ms=3),
        replace(benchmark, error_rate=0.1),
        replace(benchmark, oom_count=1),
        replace(benchmark, ttft_p95_ms=float("nan")),
        replace(benchmark, error_rate=-1),
    ]:
        evaluation = evaluate_slo(result, slo)
        assert not evaluation.feasible and evaluation.reasons
        assert filter_slo([benchmark, result], slo) == [benchmark]
    assert evaluate_slo(
        replace(benchmark, oom_count=1), SLO(10, 2, require_zero_oom=False)
    ).feasible
    with pytest.raises(ValueError):
        evaluate_slo(benchmark, SLO(-1, 2))


def test_pareto(benchmark: BenchmarkResult) -> None:
    dominated = replace(benchmark, config_id="dominated", ttft_p95_ms=11)
    tradeoff = replace(
        benchmark,
        config_id="tradeoff",
        ttft_p95_ms=11,
        output_tokens_per_second_per_gpu=110,
    )
    tie = replace(benchmark, config_id="tie")
    missing = replace(benchmark, ttft_p95_ms=None)
    invalid = replace(benchmark, tpot_p95_ms=float("inf"))
    assert pareto_frontier([dominated, benchmark, tradeoff, tie, missing, invalid]) == [
        benchmark,
        tradeoff,
        tie,
    ]
    assert pareto_frontier([]) == []


def test_recommendation(
    stats: list[ExpertStats], topology: TopologyGraph, benchmark: BenchmarkResult
) -> None:
    c = PlacementConstraints([0, 1, 2, 3])
    prediction = recommend(stats, topology, c)
    assert prediction.benchmark_slo is None
    result = recommend(stats, topology, c, benchmark=benchmark, slo=SLO(10, 2))
    assert result.benchmark_slo is not None and result.benchmark_slo.feasible
    assert result.benchmark_config_id == benchmark.config_id
    assert "predictions" in " ".join(result.explanation)
    with pytest.raises(ValueError):
        recommend(stats, topology, c, benchmark=benchmark)
