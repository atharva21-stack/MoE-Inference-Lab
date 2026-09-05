"""Stable public API surface for the topology/optimizer lane.

See CLAUDE_MOEFORGE.md section 8 ("API Surface"). This module intentionally
contains no logic of its own — it re-exports the already-tested functions
from moeforge.topology and moeforge.optimizer so dashboard code and the
Codex lane have one small, stable import surface instead of reaching into
optimizer/topology internals directly.
"""

from moeforge.optimizer.load_model import calculate_expert_load
from moeforge.optimizer.pareto import pareto_frontier
from moeforge.optimizer.placement import optimize_placement
from moeforge.topology.parser import parse_topology

__all__ = [
    "calculate_expert_load",
    "optimize_placement",
    "pareto_frontier",
    "parse_topology",
]
