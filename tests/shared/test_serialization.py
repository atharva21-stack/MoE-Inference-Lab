"""Synthetic contract serialization checks for optimizer integration."""

import json

import pytest

from moeforge.shared.models import (
    BenchmarkResult,
    ExpertStats,
    PlacementPlan,
    TopologyGraph,
)
from moeforge.shared.serialization import to_json


def test_roundtrip(benchmark: BenchmarkResult, stats: list[ExpertStats]) -> None:
    assert BenchmarkResult(**json.loads(to_json(benchmark))) == benchmark
    assert ExpertStats(**json.loads(to_json(stats[0]))) == stats[0]


def test_graph_plan(topology: TopologyGraph) -> None:
    assert len(json.loads(to_json(topology))["links"]) == 16
    plan = PlacementPlan({0: 1}, 1.0, None, True)
    assert json.loads(to_json(plan))["expert_to_gpu"] == {"0": 1}
    assert json.loads(to_json(plan))["predicted_communication_cost"] is None


def test_nonfinite_rejected() -> None:
    with pytest.raises(ValueError):
        to_json(PlacementPlan({}, float("nan"), None, True))
