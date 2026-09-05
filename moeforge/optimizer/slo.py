"""Conservative SLO evaluation: unavailable required metrics never pass."""

import math

from moeforge.shared.models import SLO, BenchmarkResult, SLOEvaluation


def evaluate_slo(result: BenchmarkResult, slo: SLO) -> SLOEvaluation:
    if (
        any(
            not math.isfinite(v) or v < 0
            for v in (
                slo.p95_ttft_ms,
                slo.p95_tpot_ms,
                slo.max_error_rate,
            )
        )
        or slo.max_error_rate > 1
    ):
        raise ValueError(
            "SLO limits must be finite and nonnegative; error rate must be <= 1"
        )
    reasons = []

    def latency(value: float | None, target: float, label: str) -> bool | None:
        if value is None:
            reasons.append(f"Required {label} measurement is missing.")
            return None
        passed = math.isfinite(value) and 0 <= value <= target
        if not passed:
            reasons.append(f"{label} is invalid or exceeds its SLO target.")
        return passed

    ttft = latency(result.ttft_p95_ms, slo.p95_ttft_ms, "P95 TTFT")
    tpot = latency(result.tpot_p95_ms, slo.p95_tpot_ms, "P95 TPOT")
    error = (
        math.isfinite(result.error_rate)
        and 0 <= result.error_rate <= slo.max_error_rate
    )
    if not error:
        reasons.append("Error rate is invalid or exceeds its SLO target.")
    oom = result.oom_count == 0 if slo.require_zero_oom else None
    if oom is False:
        reasons.append("Zero OOM events are required.")
    return SLOEvaluation(
        ttft is True and tpot is True and error and oom is not False,
        ttft,
        tpot,
        error,
        oom,
        reasons,
    )


def filter_slo(candidates: list[BenchmarkResult], slo: SLO) -> list[BenchmarkResult]:
    return [
        candidate for candidate in candidates if evaluate_slo(candidate, slo).feasible
    ]
