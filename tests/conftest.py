"""Synthetic fixtures only: these values are not inference measurements."""

import pytest

from moeforge.shared.models import BenchmarkResult, ExpertStats, TopologyGraph
from moeforge.topology.parser import parse_topology


@pytest.fixture
def topology() -> TopologyGraph:
    return parse_topology("""
        GPU0 GPU1 GPU2 GPU3
GPU0 X NV4 SYS SYS
GPU1 NV4 X SYS SYS
GPU2 SYS SYS X NV4
GPU3 SYS SYS NV4 X
""")


@pytest.fixture
def stats() -> list[ExpertStats]:
    return [
        ExpertStats(0, e, e // 2, load)
        for e, load in enumerate([100, 120, 900, 110, 220, 180, 140, 130])
    ]


@pytest.fixture
def benchmark() -> BenchmarkResult:
    return BenchmarkResult(
        experiment_id="synthetic-test",
        config_id="synthetic-baseline",
        gpu_count=4,
        request_count=100,
        successful_requests=100,
        failed_requests=0,
        ttft_p50_ms=5,
        ttft_p95_ms=10,
        ttft_p99_ms=12,
        tpot_p50_ms=1,
        tpot_p95_ms=2,
        tpot_p99_ms=3,
        e2e_p50_ms=20,
        e2e_p95_ms=30,
        e2e_p99_ms=40,
        input_tokens_per_second=200,
        output_tokens_per_second=400,
        total_tokens_per_second=600,
        requests_per_second=10,
        output_tokens_per_second_per_gpu=100,
        error_rate=0,
    )
