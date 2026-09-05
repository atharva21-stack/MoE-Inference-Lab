"""Non-dominated measured candidates; incomplete/invalid objectives are excluded."""

import math

from moeforge.shared.models import BenchmarkResult


def pareto_frontier(candidates: list[BenchmarkResult]) -> list[BenchmarkResult]:
    points: list[tuple[BenchmarkResult, tuple[float, float, float]]] = []
    for candidate in candidates:
        a, b, c = (
            candidate.ttft_p95_ms,
            candidate.tpot_p95_ms,
            candidate.output_tokens_per_second_per_gpu,
        )
        if a is None or b is None or c is None or candidate.gpu_count <= 0:
            continue
        if not all(math.isfinite(v) and v >= 0 for v in (a, b, c)):
            continue
        points.append((candidate, (a, b, -c)))
    return [
        candidate
        for candidate, point in points
        if not any(
            all(x <= y for x, y in zip(other, point, strict=True))
            and any(x < y for x, y in zip(other, point, strict=True))
            for _, other in points
        )
    ]
