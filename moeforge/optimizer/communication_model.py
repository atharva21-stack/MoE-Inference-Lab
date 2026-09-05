"""V1 token-weighted relative cost. No conversion to microseconds."""

import math

from moeforge.shared.models import ExpertTraffic, TopologyGraph


def communication_cost(traffic: list[ExpertTraffic], topology: TopologyGraph) -> float:
    total = 0.0
    for item in traffic:
        if item.token_count < 0:
            raise ValueError("Traffic counts must be nonnegative")
        if (
            item.source_gpu not in topology.gpus
            or item.destination_gpu not in topology.gpus
        ):
            raise ValueError("Traffic references an unknown GPU")
        try:
            cost = topology.links[item.source_gpu, item.destination_gpu].relative_cost
        except KeyError as exc:
            raise ValueError("Missing topology link") from exc
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid relative link cost")
        total += item.token_count * cost
    return total
