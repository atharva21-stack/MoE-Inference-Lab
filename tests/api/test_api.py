"""moeforge.api is a thin re-export layer; these are smoke tests only.

Behavioral coverage for each function lives in tests/topology and
tests/optimizer against the underlying implementations.
"""

from moeforge.api import (
    calculate_expert_load,
    optimize_placement,
    pareto_frontier,
    parse_topology,
)
from moeforge.shared.models import PlacementConstraints


def test_parse_topology_reexport_matches_direct_import():
    from moeforge.topology.parser import parse_topology as direct

    assert parse_topology is direct


def test_calculate_expert_load_reexport_matches_direct_import():
    from moeforge.optimizer.load_model import calculate_expert_load as direct

    assert calculate_expert_load is direct


def test_optimize_placement_reexport_matches_direct_import():
    from moeforge.optimizer.placement import optimize_placement as direct

    assert optimize_placement is direct


def test_pareto_frontier_reexport_matches_direct_import():
    from moeforge.optimizer.pareto import pareto_frontier as direct

    assert pareto_frontier is direct


def test_end_to_end_smoke(topology, stats, benchmark):
    summary = calculate_expert_load(stats)
    assert summary.hot_experts == [2]

    plan = optimize_placement(
        stats,
        topology,
        PlacementConstraints(gpu_ids=[0, 1, 2, 3]),
    )
    assert plan.feasible
    assert set(plan.expert_to_gpu) == {s.expert_id for s in stats}

    assert pareto_frontier([benchmark]) == [benchmark]


def test_parse_topology_from_raw_text():
    graph = parse_topology("""
        GPU0 GPU1
GPU0 X NV4
GPU1 NV4 X
""")
    assert graph.cost(0, 1) == 1.5


def test_calculate_expert_load_empty_input():
    summary = calculate_expert_load([])
    assert summary.hot_experts == []
    assert summary.mean_expert_load == 0.0
