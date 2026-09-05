"""Contract tests for moeforge/shared/models.py (INTERFACES.md section 26).

Both implementation lanes must keep these passing.
"""

from moeforge.shared.models import (
    BenchmarkResult,
    ExpertStats,
    GPUNode,
    PlacementPlan,
    TopologyGraph,
    TopologyLink,
    to_json_dict,
)


def test_benchmark_result_construction():
    result = BenchmarkResult(
        experiment_id="exp-1",
        config_id="cfg-linear",
        gpu_count=4,
        request_count=100,
        successful_requests=99,
        failed_requests=1,
        ttft_p50_ms=100.0,
        ttft_p95_ms=200.0,
        ttft_p99_ms=250.0,
        tpot_p50_ms=20.0,
        tpot_p95_ms=25.0,
        tpot_p99_ms=30.0,
        e2e_p50_ms=500.0,
        e2e_p95_ms=900.0,
        e2e_p99_ms=1200.0,
        input_tokens_per_second=1000.0,
        output_tokens_per_second=4000.0,
        total_tokens_per_second=5000.0,
        requests_per_second=10.0,
        output_tokens_per_second_per_gpu=1000.0,
        error_rate=0.01,
    )
    assert result.gpu_count == 4
    assert result.oom_count == 0


def test_expert_stats_construction():
    stats = ExpertStats(layer_id=0, expert_id=7, gpu_id=1, routed_tokens=900)
    assert stats.mean_expert_load is None
    assert stats.imbalance_ratio is None


def test_topology_graph_construction_and_cost():
    gpus = {0: GPUNode(gpu_id=0), 1: GPUNode(gpu_id=1)}
    links = {
        (0, 1): TopologyLink(src_gpu=0, dst_gpu=1, link_type="NV4", relative_cost=1.5),
        (1, 0): TopologyLink(src_gpu=1, dst_gpu=0, link_type="NV4", relative_cost=1.5),
    }
    graph = TopologyGraph(gpus=gpus, links=links)

    assert graph.cost(0, 1) == 1.5
    assert graph.cost(0, 0) == 0.0
    assert graph.get_link(0, 1).link_type == "NV4"


def test_placement_plan_construction():
    plan = PlacementPlan(
        expert_to_gpu={0: 0, 1: 1},
        predicted_imbalance=1.2,
        predicted_communication_cost=42.0,
        feasible=True,
        explanation=["balanced"],
    )
    assert plan.feasible is True
    assert plan.expert_to_gpu[1] == 1


def test_json_serialization_stringifies_tuple_keys():
    gpus = {0: GPUNode(gpu_id=0)}
    links = {
        (0, 0): TopologyLink(src_gpu=0, dst_gpu=0, link_type="X", relative_cost=0.0)
    }
    graph = TopologyGraph(gpus=gpus, links=links)

    payload = to_json_dict(graph)

    assert "(0, 0)" in payload["links"]
    assert payload["links"]["(0, 0)"]["link_type"] == "X"
