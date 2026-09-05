"""Relative optimizer costs, never bandwidth or latency claims."""

import math

LINK_COST: dict[str, float] = {
    "X": 0.0,
    "NV18": 1.0,
    "NV12": 1.1,
    "NV8": 1.2,
    "NV4": 1.5,
    "NV2": 2.0,
    "NV1": 2.5,
    "PIX": 4.0,
    "PXB": 5.0,
    "PHB": 7.0,
    "NODE": 8.0,
    "SYS": 10.0,
}


def link_cost(label: str, overrides: dict[str, float] | None = None) -> float:
    costs = LINK_COST | (overrides or {})
    if label not in costs:
        raise ValueError(
            f"Unknown topology label {label!r}; supply an explicit cost override"
        )
    value = costs[label]
    if not math.isfinite(value) or value < 0:
        raise ValueError("Link costs must be finite and nonnegative")
    if label == "X" and value != 0:
        raise ValueError("Local link cost must be zero")
    return value
