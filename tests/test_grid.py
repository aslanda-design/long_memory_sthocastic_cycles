import numpy as np
import pytest

from cyclical_fractional_test import InvalidConfigurationError, StochasticCycle
from cyclical_fractional_test import CyclicalTestConfig
from cyclical_fractional_test.grid import (
    build_d_fine_grid,
    build_d_grid,
    build_d_grid_for_strategy,
    build_default_d_coarse_grid,
    build_multi_cycle_candidate_grid,
    build_multi_cycle_candidate_grid_from_d_grids,
    build_r_grid_around_peak,
    build_single_cycle_candidate_grid,
    candidate_iterator,
)


# ---------------------------------------------------------------------------
# build_r_grid_around_peak
# ---------------------------------------------------------------------------


def test_build_r_grid_around_peak_standard_case():
    result = build_r_grid_around_peak(r_peak=25, r_window=10, T=100)
    expected = np.arange(15, 36)
    np.testing.assert_array_equal(result, expected)


def test_build_r_grid_around_peak_clips_lower_bound():
    result = build_r_grid_around_peak(r_peak=5, r_window=10, T=100)
    expected = np.arange(1, 16)
    np.testing.assert_array_equal(result, expected)


def test_build_r_grid_around_peak_clips_upper_bound():
    result = build_r_grid_around_peak(r_peak=95, r_window=10, T=100)
    expected = np.arange(85, 100)
    np.testing.assert_array_equal(result, expected)


def test_build_r_grid_zero_window_returns_single_element():
    result = build_r_grid_around_peak(r_peak=25, r_window=0, T=100)
    np.testing.assert_array_equal(result, np.array([25]))


def test_build_r_grid_clips_both_bounds():
    result = build_r_grid_around_peak(r_peak=1, r_window=5, T=4)
    # r_min = max(1, -4) = 1; r_max = min(3, 6) = 3
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_build_r_grid_can_include_zero_when_requested():
    result = build_r_grid_around_peak(
        r_peak=0, r_window=2, T=10, include_zero=True
    )
    np.testing.assert_array_equal(result, np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# build_d_grid
# ---------------------------------------------------------------------------


def test_build_d_grid_default():
    result = build_d_grid()
    assert len(result) == 11
    np.testing.assert_allclose(result, np.linspace(0.0, 1.0, 11))


def test_build_d_grid_default_starts_at_zero_ends_at_one():
    result = build_d_grid()
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_build_d_grid_custom():
    result = build_d_grid([0.0, 0.25, 0.5])
    np.testing.assert_allclose(result, np.array([0.0, 0.25, 0.5]))


def test_build_d_grid_custom_numpy_input():
    arr = np.array([0.0, 0.5, 1.0])
    result = build_d_grid(arr)
    np.testing.assert_allclose(result, arr)


def test_build_d_grid_returns_ndarray():
    assert isinstance(build_d_grid(), np.ndarray)
    assert isinstance(build_d_grid([0.1, 0.5]), np.ndarray)


# ---------------------------------------------------------------------------
# build_default_d_coarse_grid / build_d_fine_grid (adaptive search)
# ---------------------------------------------------------------------------


def test_default_coarse_grid_is_exact():
    result = build_default_d_coarse_grid()
    np.testing.assert_allclose(result, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))


def test_fine_grid_centered_at_03():
    result = build_d_fine_grid(0.3, radius=0.09, step=0.01)
    expected = np.array([round(0.21 + 0.01 * i, 12) for i in range(19)])
    assert len(result) == 19
    np.testing.assert_allclose(result, expected)
    assert result[0] == pytest.approx(0.21)
    assert result[-1] == pytest.approx(0.39)


def test_fine_grid_clipped_at_lower_boundary():
    result = build_d_fine_grid(0.0, radius=0.09, step=0.01)
    expected = np.array([round(0.01 * i, 12) for i in range(10)])
    np.testing.assert_allclose(result, expected)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(0.09)


def test_fine_grid_clipped_at_upper_boundary():
    result = build_d_fine_grid(1.0, radius=0.09, step=0.01)
    expected = np.array([round(0.91 + 0.01 * i, 12) for i in range(10)])
    np.testing.assert_allclose(result, expected)
    assert result[0] == pytest.approx(0.91)
    assert result[-1] == pytest.approx(1.0)


def test_fine_grid_values_stay_in_unit_interval():
    result = build_d_fine_grid(0.95, radius=0.09, step=0.01)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_build_d_grid_for_strategy_adaptive_returns_coarse():
    cfg = CyclicalTestConfig(d_search_strategy="adaptive")
    np.testing.assert_allclose(build_d_grid_for_strategy(cfg), build_default_d_coarse_grid())


def test_build_d_grid_for_strategy_fixed_uses_d_grid():
    cfg = CyclicalTestConfig(d_search_strategy="fixed_grid", d_grid=np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(build_d_grid_for_strategy(cfg), np.array([0.0, 0.5, 1.0]))


def test_build_d_grid_for_strategy_adaptive_custom_coarse():
    cfg = CyclicalTestConfig(d_search_strategy="adaptive", d_coarse_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(build_d_grid_for_strategy(cfg), np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


# ---------------------------------------------------------------------------
# build_single_cycle_candidate_grid
# ---------------------------------------------------------------------------


def test_build_single_cycle_candidate_grid_count():
    r_grid = np.array([10, 11])
    d_grid = np.array([0.0, 0.5, 1.0])
    candidates = build_single_cycle_candidate_grid(r_grid, d_grid)
    assert len(candidates) == 6


def test_build_single_cycle_candidate_grid_returns_stochastic_cycles():
    r_grid = np.array([10, 11])
    d_grid = np.array([0.0, 0.5])
    candidates = build_single_cycle_candidate_grid(r_grid, d_grid)
    for c in candidates:
        assert isinstance(c, tuple)
        assert len(c) == 1
        assert isinstance(c[0], StochasticCycle)


def test_build_single_cycle_candidate_grid_r_and_d_values():
    r_grid = np.array([5])
    d_grid = np.array([0.3])
    candidates = build_single_cycle_candidate_grid(r_grid, d_grid)
    assert len(candidates) == 1
    assert candidates[0][0].R == 5
    assert candidates[0][0].D == pytest.approx(0.3)


def test_build_single_cycle_candidate_grid_cartesian_order():
    r_grid = np.array([10, 11])
    d_grid = np.array([0.0, 1.0])
    candidates = build_single_cycle_candidate_grid(r_grid, d_grid)
    assert candidates[0][0].R == 10 and candidates[0][0].D == pytest.approx(0.0)
    assert candidates[1][0].R == 10 and candidates[1][0].D == pytest.approx(1.0)
    assert candidates[2][0].R == 11 and candidates[2][0].D == pytest.approx(0.0)
    assert candidates[3][0].R == 11 and candidates[3][0].D == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_multi_cycle_candidate_grid
# ---------------------------------------------------------------------------


def test_build_multi_cycle_candidate_grid_builds_joint_d_product():
    r_grid = np.array([2, 5])
    d_grid = np.array([0.0, 0.5])
    candidates = build_multi_cycle_candidate_grid(r_grid, d_grid)
    assert len(candidates) == 4
    assert candidates == [
        (StochasticCycle(R=2, D=0.0), StochasticCycle(R=5, D=0.0)),
        (StochasticCycle(R=2, D=0.0), StochasticCycle(R=5, D=0.5)),
        (StochasticCycle(R=2, D=0.5), StochasticCycle(R=5, D=0.0)),
        (StochasticCycle(R=2, D=0.5), StochasticCycle(R=5, D=0.5)),
    ]


def test_build_multi_cycle_candidate_grid_from_d_grids_builds_joint_product():
    r_grid = np.array([2, 5])
    d_grids = [np.array([0.0, 0.5]), np.array([0.25, 0.75])]
    candidates = build_multi_cycle_candidate_grid_from_d_grids(r_grid, d_grids)
    assert len(candidates) == 4
    assert candidates == [
        (StochasticCycle(R=2, D=0.0), StochasticCycle(R=5, D=0.25)),
        (StochasticCycle(R=2, D=0.0), StochasticCycle(R=5, D=0.75)),
        (StochasticCycle(R=2, D=0.5), StochasticCycle(R=5, D=0.25)),
        (StochasticCycle(R=2, D=0.5), StochasticCycle(R=5, D=0.75)),
    ]


@pytest.mark.parametrize(
    "r_grid,d_grids",
    [
        (np.array([2, 5]), [np.array([0.0])]),
        (np.array([2, 5]), None),
        (np.array([2, 5]), [np.array([]), np.array([0.0])]),
        (np.array([2, 5]), [np.array([0.0]), np.array([1.5])]),
    ],
)
def test_build_multi_cycle_candidate_grid_from_d_grids_rejects_invalid(
    r_grid, d_grids
):
    with pytest.raises(InvalidConfigurationError):
        build_multi_cycle_candidate_grid_from_d_grids(r_grid, d_grids)


# ---------------------------------------------------------------------------
# candidate_iterator
# ---------------------------------------------------------------------------


def test_candidate_iterator_single_mode():
    r_grid = np.array([10, 11])
    d_grid = np.array([0.0, 0.5])
    candidates = list(candidate_iterator(r_grid, d_grid, stochastic_cycle_mode="single"))
    assert len(candidates) == 4
    for c in candidates:
        assert isinstance(c, tuple)
        assert len(c) == 1


def test_candidate_iterator_multi_peak_single_cycle_mode():
    r_grid = np.array([10, 11])
    d_grid = np.array([0.0, 0.5])
    candidates = list(
        candidate_iterator(r_grid, d_grid, stochastic_cycle_mode="multi_peak_single_cycle")
    )
    assert len(candidates) == 4


def test_candidate_iterator_multi_cycle_mode():
    r_grid = np.array([10, 12])
    d_grid = np.array([0.0, 0.5])
    candidates = candidate_iterator(
        r_grid, d_grid, stochastic_cycle_mode="multi_cycle"
    )
    assert len(candidates) == 4
    assert all(len(candidate) == 2 for candidate in candidates)


def test_candidate_iterator_rejects_invalid_mode():
    with pytest.raises(InvalidConfigurationError):
        candidate_iterator(
            np.array([10]), np.array([0.5]), stochastic_cycle_mode="unknown_mode"
        )


# ---------------------------------------------------------------------------
# Error cases — build_r_grid_around_peak
# ---------------------------------------------------------------------------


def test_build_r_grid_rejects_r_peak_zero():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=0, r_window=5, T=100)


def test_build_r_grid_rejects_r_peak_equals_T():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=100, r_window=0, T=100)


def test_build_r_grid_rejects_r_peak_float():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=25.5, r_window=5, T=100)  # type: ignore


def test_build_r_grid_rejects_negative_r_window():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=25, r_window=-1, T=100)


def test_build_r_grid_rejects_T_less_than_2():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=1, r_window=0, T=1)


def test_build_r_grid_rejects_float_T():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=25, r_window=5, T=100.0)  # type: ignore


def test_build_r_grid_rejects_non_boolean_include_zero():
    with pytest.raises(InvalidConfigurationError):
        build_r_grid_around_peak(r_peak=0, r_window=1, T=10, include_zero=1)  # type: ignore


# ---------------------------------------------------------------------------
# Error cases — build_d_grid
# ---------------------------------------------------------------------------


def test_build_d_grid_rejects_values_below_zero():
    with pytest.raises(InvalidConfigurationError):
        build_d_grid([-0.1, 0.5])


def test_build_d_grid_rejects_values_above_one():
    with pytest.raises(InvalidConfigurationError):
        build_d_grid([0.5, 1.2])


def test_build_d_grid_rejects_nan():
    with pytest.raises(InvalidConfigurationError):
        build_d_grid([0.1, float("nan")])


def test_build_d_grid_rejects_empty():
    with pytest.raises(InvalidConfigurationError):
        build_d_grid([])
