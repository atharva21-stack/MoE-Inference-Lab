import pytest

from moeforge.optimizer.load_model import calculate_expert_load
from moeforge.shared.models import ExpertStats


def test_load(stats: list[ExpertStats]) -> None:
    result = calculate_expert_load(stats)
    assert result.total_routed_tokens == 1900
    assert result.mean_expert_load == 237.5
    assert result.max_expert_load == 900
    assert result.imbalance_ratio == pytest.approx(900 / 237.5)
    assert result.balancedness == pytest.approx(237.5 / 900)
    assert result.hot_experts == [2]
    assert result.tokens_by_gpu == {0: 220, 1: 1010, 2: 400, 3: 270}
    assert calculate_expert_load(stats, hot_threshold=700).hot_experts == []


def test_empty_zero_and_aggregation() -> None:
    for stats in [[], [ExpertStats(0, 0, 0, 0)]]:
        result = calculate_expert_load(stats)
        assert result.imbalance_ratio == 0
        assert result.balancedness == 1
    result = calculate_expert_load([ExpertStats(0, 0, 0, 2), ExpertStats(0, 0, 1, 3)])
    assert result.tokens_by_expert == {0: 5}


def test_invalid() -> None:
    with pytest.raises(ValueError):
        calculate_expert_load([ExpertStats(0, 0, 0, -1)])
    with pytest.raises(ValueError, match="one layer"):
        calculate_expert_load([ExpertStats(0, 0, 0, 1), ExpertStats(1, 0, 0, 1)])
