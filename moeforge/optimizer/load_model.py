"""Load analysis for one layer; include zero-load experts in the input."""

import math

from moeforge.shared.models import ExpertLoadSummary, ExpertStats


def calculate_expert_load(
    stats: list[ExpertStats],
    *,
    hot_threshold: float = 0.0,
) -> ExpertLoadSummary:
    """Hot means strictly greater than mean + hot_threshold tokens.

    Repeated records are summed (caller must supply non-overlapping windows).
    Different layers are rejected because placement keys are bare expert IDs.
    """
    if not math.isfinite(hot_threshold) or hot_threshold < 0:
        raise ValueError("Hot threshold must be finite and nonnegative")
    if len({s.layer_id for s in stats}) > 1:
        raise ValueError("Analyze one layer at a time")
    experts: dict[int, int] = {}
    gpus: dict[int, int] = {}
    for s in stats:
        if min(s.layer_id, s.expert_id, s.gpu_id, s.routed_tokens) < 0:
            raise ValueError("IDs and token counts must be nonnegative")
        experts[s.expert_id] = experts.get(s.expert_id, 0) + s.routed_tokens
        gpus[s.gpu_id] = gpus.get(s.gpu_id, 0) + s.routed_tokens
    total = sum(experts.values())
    mean = total / len(experts) if experts else 0.0
    peak = max(experts.values(), default=0)
    return ExpertLoadSummary(
        total,
        mean,
        peak,
        peak / mean if mean else 0.0,
        mean / peak if peak else 1.0,
        dict(sorted(experts.items())),
        dict(sorted(gpus.items())),
        sorted(e for e, load in experts.items() if load > mean + hot_threshold),
    )
