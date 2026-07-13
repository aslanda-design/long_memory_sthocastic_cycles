import numpy as np
import pytest

from cyclical_fractional_test.exceptions import InvalidConfigurationError
from cyclical_fractional_test.results import GridCandidateResult, StochasticCycle
from cyclical_fractional_test.scoring import (
    TopKSelector,
    compute_test_star_statistic,
    compute_test_statistic,
    score_candidate,
)


def _candidate(test_value=None, test_star_value=None, R=2, D=0.4):
    return GridCandidateResult(
        cycles=(StochasticCycle(R=R, D=D),),
        test_value=test_value,
        test_star_value=test_star_value,
    )


# ---------------------------------------------------------------------------
# compute_test_statistic
# ---------------------------------------------------------------------------


def test_compute_test_statistic_known_value():
    result = compute_test_statistic(T=100, xa=2.0, xaa=4.0, variance_time=5.0)
    expected = (np.sqrt(100) / np.sqrt(4.0)) * (2.0 / 5.0)
    assert np.isclose(result, expected)


def test_compute_test_statistic_preserves_sign():
    assert compute_test_statistic(T=100, xa=2.0, xaa=4.0, variance_time=5.0) > 0
    assert compute_test_statistic(T=100, xa=-2.0, xaa=4.0, variance_time=5.0) < 0


def test_compute_test_statistic_zero_xa():
    assert compute_test_statistic(T=100, xa=0.0, xaa=4.0, variance_time=5.0) == 0.0


@pytest.mark.parametrize("bad_xaa", [0.0, -1.0, float("nan"), float("inf")])
def test_compute_test_statistic_rejects_invalid_xaa(bad_xaa):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_test_statistic(T=100, xa=2.0, xaa=bad_xaa, variance_time=5.0)


@pytest.mark.parametrize("bad_var", [0.0, -1.0, float("nan"), float("inf")])
def test_compute_test_statistic_rejects_invalid_variance(bad_var):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_test_statistic(T=100, xa=2.0, xaa=4.0, variance_time=bad_var)


# ---------------------------------------------------------------------------
# compute_test_star_statistic
# ---------------------------------------------------------------------------


def test_compute_test_star_statistic_known_value():
    result = compute_test_star_statistic(T=100, xa=2.0, xaa=4.0, variance_frequency=10.0)
    expected = (np.sqrt(100) / np.sqrt(4.0)) * (2.0 / 10.0)
    assert np.isclose(result, expected)


def test_compute_test_star_statistic_preserves_sign():
    assert compute_test_star_statistic(T=100, xa=-2.0, xaa=4.0, variance_frequency=10.0) < 0


@pytest.mark.parametrize("bad_var", [0.0, -1.0, float("nan"), float("inf")])
def test_compute_test_star_statistic_rejects_invalid_variance(bad_var):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_test_star_statistic(T=100, xa=2.0, xaa=4.0, variance_frequency=bad_var)


def test_compute_test_star_statistic_rejects_xaa_zero():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_test_star_statistic(T=100, xa=2.0, xaa=0.0, variance_frequency=10.0)


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


def test_score_candidate_standard_uses_abs_test_value():
    assert np.isclose(score_candidate(_candidate(test_value=-0.25, test_star_value=10.0), "standard"), 0.25)


def test_score_candidate_frequency_uses_abs_test_star_value():
    assert np.isclose(score_candidate(_candidate(test_value=0.25, test_star_value=-0.1), "frequency"), 0.1)


def test_score_candidate_accepts_mode_aliases():
    c = _candidate(test_value=-0.7, test_star_value=-0.5)
    assert np.isclose(score_candidate(c, "test"), 0.7)
    assert np.isclose(score_candidate(c, "test_star"), 0.5)


def test_score_candidate_rejects_unknown_mode():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        score_candidate(_candidate(test_value=0.25, test_star_value=0.1), "unknown")


def test_score_candidate_rejects_none_values():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        score_candidate(_candidate(test_value=None, test_star_value=0.1), "standard")


# ---------------------------------------------------------------------------
# TopKSelector
# ---------------------------------------------------------------------------


def test_top_k_selector_keeps_k_best():
    selector = TopKSelector(k=3, statistic_mode="standard")
    for tv in [5.0, -0.2, 1.0, 0.1, -3.0]:
        selector.consider(_candidate(test_value=tv, test_star_value=0.0))
    top = selector.get_top_k()
    assert len(top) == 3
    assert np.allclose(sorted(abs(c.test_value) for c in top), [0.1, 0.2, 1.0])


def test_top_k_selector_ordered_ascending_by_abs():
    selector = TopKSelector(k=5, statistic_mode="standard")
    for tv in [5.0, -0.2, 1.0, 0.1, -3.0]:
        selector.consider(_candidate(test_value=tv, test_star_value=0.0))
    scores = [abs(c.test_value) for c in selector.get_top_k()]
    assert scores == sorted(scores)


def test_top_k_selector_get_best_returns_smallest_score():
    selector = TopKSelector(k=3, statistic_mode="standard")
    for tv in [5.0, -0.2, 1.0, 0.1, -3.0]:
        selector.consider(_candidate(test_value=tv, test_star_value=0.0))
    assert np.isclose(abs(selector.get_best().test_value), 0.1)


def test_top_k_selector_frequency_mode():
    selector = TopKSelector(k=2, statistic_mode="frequency")
    for tsv in [2.0, -0.5, 0.1]:
        selector.consider(_candidate(test_value=0.0, test_star_value=tsv))
    scores = [abs(c.test_star_value) for c in selector.get_top_k()]
    assert np.allclose(scores, [0.1, 0.5])


def test_top_k_selector_empty_returns_none_and_empty_list():
    selector = TopKSelector(k=3, statistic_mode="standard")
    assert selector.get_best() is None
    assert selector.get_top_k() == []


def test_top_k_selector_fewer_than_k_returns_all():
    selector = TopKSelector(k=10, statistic_mode="standard")
    selector.consider(_candidate(test_value=0.5))
    selector.consider(_candidate(test_value=-1.5))
    top = selector.get_top_k()
    assert len(top) == 2
    assert np.isclose(abs(top[0].test_value), 0.5)


def test_top_k_selector_accepts_test_aliases():
    selector = TopKSelector(k=2, statistic_mode="test")
    for tv in [0.5, -2.0, 0.1]:
        selector.consider(_candidate(test_value=tv, test_star_value=99.0))
    assert np.allclose(sorted(abs(c.test_value) for c in selector.get_top_k()), [0.1, 0.5])


@pytest.mark.parametrize("bad_k", [0, -1, 1.5, True])
def test_top_k_selector_rejects_invalid_k(bad_k):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        TopKSelector(k=bad_k, statistic_mode="standard")


def test_top_k_selector_rejects_unknown_mode():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        TopKSelector(k=3, statistic_mode="unknown")
