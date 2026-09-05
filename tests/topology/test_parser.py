import pytest

from moeforge.shared.models import TopologyGraph
from moeforge.topology.parser import parse_topology


def test_links(topology: TopologyGraph) -> None:
    assert len(topology.gpus) == 4
    assert len(topology.links) == 16
    assert topology.links[0, 1].relative_cost == 1.5
    assert topology.links[0, 2].relative_cost == 10
    assert topology.links[0, 0].relative_cost == 0
    assert topology.links[0, 1].measured_bandwidth_gbps is None


def test_affinity_nic_columns() -> None:
    graph = parse_topology("""
GPU0 GPU2 NIC0 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0 X PHB PIX 0-31 0 N/A
GPU2 PHB X PIX 32-63 1 N/A
NIC0 PIX PIX X
Legend:
GPU0 irrelevant
""")
    assert set(graph.gpus) == {0, 2}
    assert graph.links[2, 0].relative_cost == 7


def test_unknown_override() -> None:
    text = "GPU0 GPU1\nGPU0 X NV99\nGPU1 NV99 X"
    with pytest.raises(ValueError, match="Unknown"):
        parse_topology(text)
    assert parse_topology(text, costs={"NV99": 0.8}).links[0, 1].relative_cost == 0.8


@pytest.mark.parametrize(
    "text",
    [
        "",
        "GPU0 GPU1\nGPU0 X NV4",
        "GPU0 GPU1\nGPU0 X\nGPU1 NV4 X",
        "GPU0 GPU1\nGPU0 X NV4\nGPU0 X NV4\nGPU1 NV4 X",
        "GPU0 GPU1\nGPU0 NV4 X\nGPU1 NV4 X",
        "GPU0 GPU0\nGPU0 X X",
    ],
)
def test_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        parse_topology(text)


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_invalid_cost(cost: float) -> None:
    with pytest.raises(ValueError):
        parse_topology("GPU0 GPU1\nGPU0 X NV4\nGPU1 NV4 X", costs={"NV4": cost})


def test_single_gpu() -> None:
    assert parse_topology("GPU0\nGPU0 X").links[0, 0].relative_cost == 0
