"""Parse GPU matrices, including tables with NIC and CPU affinity columns."""

import re

from moeforge.shared.models import GPUNode, TopologyGraph, TopologyLink
from moeforge.topology.cost_model import link_cost


def parse_topology(
    text: str, *, costs: dict[str, float] | None = None
) -> TopologyGraph:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    header: list[str] | None = None
    rows: dict[str, list[str]] = {}
    for parts in lines:
        if header is None:
            if (
                re.fullmatch(r"GPU\d+", parts[0])
                and len(parts) > 1
                and (re.fullmatch(r"GPU\d+", parts[1]) or parts[1] in {"CPU", "NIC0"})
            ):
                header = parts
            continue
        if parts[0] == "Legend:":
            break
        if re.fullmatch(r"GPU\d+", parts[0]):
            if parts[0] in rows:
                raise ValueError(f"Duplicate GPU row: {parts[0]}")
            rows[parts[0]] = parts[1:]
    if header is None:
        # A single-GPU matrix can have a one-column header.
        if lines and len(lines[0]) == 1 and re.fullmatch(r"GPU\d+", lines[0][0]):
            header = lines[0]
            rows = {p[0]: p[1:] for p in lines[1:] if p[0] == header[0]}
        else:
            raise ValueError("No GPU topology header found")
    columns = [
        (i, label) for i, label in enumerate(header) if re.fullmatch(r"GPU\d+", label)
    ]
    labels = [label for _, label in columns]
    if len(set(labels)) != len(labels) or set(rows) != set(labels):
        raise ValueError("GPU rows and unique header columns must match")
    gpus = {int(label[3:]): GPUNode(int(label[3:])) for label in labels}
    links: dict[tuple[int, int], TopologyLink] = {}
    for src in labels:
        for col, dst in columns:
            if col >= len(rows[src]):
                raise ValueError(f"Incomplete GPU row: {src}")
            label = rows[src][col]
            if (src == dst) != (label == "X"):
                raise ValueError("X is required exactly on the GPU diagonal")
            a, b = int(src[3:]), int(dst[3:])
            links[a, b] = TopologyLink(a, b, label, link_cost(label, costs))
    return TopologyGraph(gpus, links)
