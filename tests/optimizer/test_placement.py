import pytest

from moeforge.optimizer.communication_model import communication_cost
from moeforge.optimizer.placement import (
    baseline_placement,
    compare_placements,
    optimize_placement,
    validate_placement,
)
from moeforge.shared.models import (
    ExpertStats,
    ExpertTraffic,
    PlacementConstraints,
    TopologyGraph,
)


def test_synthetic_comparison(
    stats: list[ExpertStats], topology: TopologyGraph
) -> None:
    c = PlacementConstraints([0, 1, 2, 3])
    linear, rr, greedy = compare_placements(stats, topology, c)
    assert linear.plan.expert_to_gpu == {e: e // 2 for e in range(8)}
    assert rr.plan.expert_to_gpu == {e: e % 4 for e in range(8)}
    assert greedy.plan.feasible
    assert greedy.max_gpu_load == 900
    assert linear.max_gpu_load == 1010
    assert rr.max_gpu_load == 1040
    assert greedy.expert_imbalance == linear.expert_imbalance
    assert greedy.communication_cost is None
    assert not validate_placement(greedy.plan.expert_to_gpu, set(range(8)), c)
    assert optimize_placement(list(reversed(stats)), topology, c) == greedy.plan


def test_memory_and_fixed(stats: list[ExpertStats], topology: TopologyGraph) -> None:
    c = PlacementConstraints(
        [0, 1, 2, 3],
        2,
        dict.fromkeys(range(4), 20),
        dict.fromkeys(range(8), 10),
        {2: 3},
    )
    plan = optimize_placement(stats, topology, c)
    assert plan.feasible
    assert plan.expert_to_gpu[2] == 3
    assert not validate_placement(plan.expert_to_gpu, set(range(8)), c)
    c.gpu_memory_budget_bytes = dict.fromkeys(range(4), 10)
    assert not optimize_placement(stats, topology, c).feasible
    c.fixed_assignments = {0: 0, 1: 0}
    assert not optimize_placement(stats, topology, c).feasible


def test_invalid_constraints(stats: list[ExpertStats], topology: TopologyGraph) -> None:
    for c in [
        PlacementConstraints([]),
        PlacementConstraints([0, 0]),
        PlacementConstraints([99]),
        PlacementConstraints([0], -1),
        PlacementConstraints([0], gpu_memory_budget_bytes={0: 10}),
        PlacementConstraints([0], fixed_assignments={99: 0}),
    ]:
        with pytest.raises(ValueError):
            optimize_placement(stats, topology, c)
    assert validate_placement({0: 99}, set(range(8)), PlacementConstraints([0]))


def test_communication(topology: TopologyGraph) -> None:
    assert (
        communication_cost([ExpertTraffic(0, 1, 10), ExpertTraffic(0, 2, 5)], topology)
        == 65
    )
    assert communication_cost([ExpertTraffic(0, 0, 10)], topology) == 0
    with pytest.raises(ValueError):
        communication_cost([ExpertTraffic(0, 1, -1)], topology)


def test_topology_changes_decision(topology: TopologyGraph) -> None:
    stats = [ExpertStats(0, 0, 0, 100)]
    c = PlacementConstraints([0, 1, 2, 3])
    sources = {(3, 0): 100}
    plan = optimize_placement(stats, topology, c, source_tokens=sources)
    assert plan.expert_to_gpu == {0: 3}
    assert plan.predicted_communication_cost == 0
    baseline = baseline_placement(stats, topology, c, "linear", source_tokens=sources)
    assert baseline.predicted_communication_cost == 1000
    assert optimize_placement(
        stats, topology, c, source_tokens=sources, topology_weight=0
    ).expert_to_gpu == {0: 0}
    with pytest.raises(ValueError, match="account for all"):
        optimize_placement(stats, topology, c, source_tokens={})


def test_empty(topology: TopologyGraph) -> None:
    plan = optimize_placement([], topology, PlacementConstraints([0, 1]))
    assert plan.feasible and plan.predicted_imbalance == 0


def test_uneven_linear_and_fixed_violation(topology: TopologyGraph) -> None:
    stats = [ExpertStats(0, e, 0, 10) for e in range(5)]
    c = PlacementConstraints([0, 1, 2, 3])
    plan = baseline_placement(stats, topology, c, "linear")
    assert plan.expert_to_gpu == {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
    c.fixed_assignments = {0: 3}
    assert not baseline_placement(stats, topology, c, "linear").feasible


def test_memory_redirect(topology: TopologyGraph) -> None:
    stats = [ExpertStats(0, 0, 0, 900), ExpertStats(0, 1, 0, 100)]
    c = PlacementConstraints(
        [0, 1], gpu_memory_budget_bytes={0: 5, 1: 10}, expert_memory_bytes={0: 10, 1: 5}
    )
    assert optimize_placement(stats, topology, c).expert_to_gpu == {0: 1, 1: 0}
