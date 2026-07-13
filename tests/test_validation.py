import numpy as np
import pytest

from cyclical_fractional_test import (
    CyclicalTestConfig,
    InvalidCycleError,
    InvalidConfigurationError,
    InvalidSeriesError,
    StochasticCycle,
)
from cyclical_fractional_test.validation import (
    validate_boolean,
    validate_chebyshev_orders,
    validate_config,
    validate_cycle,
    validate_cycles,
    validate_d_coarse_grid,
    validate_d_fine_radius,
    validate_d_fine_step,
    validate_d_grid,
    validate_ignored_stochastic_rs,
    validate_mode,
    validate_n_deterministic_cycles,
    validate_n_stochastic_cycles,
    validate_r_window,
    validate_series,
    validate_top_k,
)


# ---------------------------------------------------------------------------
# validate_series
# ---------------------------------------------------------------------------


def test_validate_series_accepts_numeric_list():
    result = validate_series([1, 2, 3, 4, 5])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (5,)


def test_validate_series_accepts_numpy_array():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = validate_series(arr)
    assert result.ndim == 1
    assert result.dtype == float


def test_validate_series_rejects_none():
    with pytest.raises(InvalidSeriesError):
        validate_series(None)


def test_validate_series_rejects_empty_array():
    with pytest.raises(InvalidSeriesError):
        validate_series([])


def test_validate_series_rejects_multidimensional_array():
    with pytest.raises(InvalidSeriesError):
        validate_series(np.array([[1, 2], [3, 4]]))


def test_validate_series_rejects_nan():
    with pytest.raises(InvalidSeriesError):
        validate_series([1, np.nan, 3, 4, 5])


def test_validate_series_rejects_inf():
    with pytest.raises(InvalidSeriesError):
        validate_series([1, np.inf, 3, 4, 5])


def test_validate_series_rejects_non_numeric_values():
    with pytest.raises(InvalidSeriesError):
        validate_series([1, "a", 3, 4, 5])


def test_validate_series_rejects_series_below_min_length():
    with pytest.raises(InvalidSeriesError):
        validate_series([1.0, 2.0])


# ---------------------------------------------------------------------------
# validate_n_deterministic_cycles
# ---------------------------------------------------------------------------


def test_validate_n_deterministic_cycles_accepts_positive_integer():
    assert validate_n_deterministic_cycles(4) == 4


def test_validate_n_deterministic_cycles_accepts_zero():
    assert validate_n_deterministic_cycles(0) == 0


def test_validate_n_deterministic_cycles_rejects_negative():
    with pytest.raises(InvalidConfigurationError):
        validate_n_deterministic_cycles(-1)


def test_validate_n_deterministic_cycles_rejects_float():
    with pytest.raises(InvalidConfigurationError):
        validate_n_deterministic_cycles(4.5)


def test_validate_chebyshev_orders_accepts_positive_orders():
    assert validate_chebyshev_orders([2, np.int64(5)]) == (2, 5)


@pytest.mark.parametrize("bad_orders", [[], [0], [-1], [2, 2], [True], [1.5]])
def test_validate_chebyshev_orders_rejects_invalid_values(bad_orders):
    with pytest.raises(InvalidConfigurationError):
        validate_chebyshev_orders(bad_orders)


def test_validate_config_accepts_explicit_chebyshev_orders():
    config = CyclicalTestConfig(chebyshev_orders=(2, 5), include_intercept=True)

    assert validate_config(config) is config


# ---------------------------------------------------------------------------
# validate_n_stochastic_cycles
# ---------------------------------------------------------------------------


def test_validate_n_stochastic_cycles_accepts_positive_integer():
    assert validate_n_stochastic_cycles(2) == 2


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, True])
def test_validate_n_stochastic_cycles_rejects_invalid_values(bad_n):
    with pytest.raises(InvalidConfigurationError):
        validate_n_stochastic_cycles(bad_n)


def test_validate_config_rejects_bad_n_stochastic_cycles():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(n_stochastic_cycles=0))


def test_validate_ignored_stochastic_rs_accepts_non_negative_indices():
    assert validate_ignored_stochastic_rs([0, np.int64(5)]) == (0, 5)


@pytest.mark.parametrize("bad_rs", [[], [-1], [2, 2], [True], [1.5]])
def test_validate_ignored_stochastic_rs_rejects_invalid_values(bad_rs):
    with pytest.raises(InvalidConfigurationError):
        validate_ignored_stochastic_rs(bad_rs)


def test_validate_config_accepts_ignored_stochastic_rs():
    config = CyclicalTestConfig(ignored_stochastic_rs=(5, 270))

    assert validate_config(config) is config


# ---------------------------------------------------------------------------
# validate_top_k
# ---------------------------------------------------------------------------


def test_validate_top_k_accepts_positive_integer():
    assert validate_top_k(5) == 5


def test_validate_top_k_rejects_zero():
    with pytest.raises(InvalidConfigurationError):
        validate_top_k(0)


def test_validate_top_k_rejects_negative():
    with pytest.raises(InvalidConfigurationError):
        validate_top_k(-3)


def test_validate_top_k_rejects_float():
    with pytest.raises(InvalidConfigurationError):
        validate_top_k(1.0)


# ---------------------------------------------------------------------------
# validate_r_window
# ---------------------------------------------------------------------------


def test_validate_r_window_accepts_zero():
    assert validate_r_window(0) == 0


def test_validate_r_window_accepts_positive_integer():
    assert validate_r_window(10) == 10


def test_validate_r_window_rejects_negative():
    with pytest.raises(InvalidConfigurationError):
        validate_r_window(-1)


def test_validate_r_window_rejects_float():
    with pytest.raises(InvalidConfigurationError):
        validate_r_window(10.0)


# ---------------------------------------------------------------------------
# validate_d_grid
# ---------------------------------------------------------------------------


def test_validate_d_grid_accepts_none():
    assert validate_d_grid(None) is None


def test_validate_d_grid_accepts_valid_grid():
    result = validate_d_grid([0.0, 0.1, 0.5, 1.0])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float


def test_validate_d_grid_rejects_empty_grid():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([])


def test_validate_d_grid_rejects_nan():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.1, np.nan, 0.5])


def test_validate_d_grid_rejects_inf():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.1, np.inf])


def test_validate_d_grid_rejects_values_below_zero():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([-0.1, 0.2])


def test_validate_d_grid_rejects_values_above_one():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.2, 1.2])


def test_validate_d_grid_rejects_non_numeric():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.1, "bad"])


# ---------------------------------------------------------------------------
# adaptive D search config (d_search_strategy, d_coarse_grid, d_fine_*)
# ---------------------------------------------------------------------------


def test_default_config_uses_adaptive_search():
    cfg = CyclicalTestConfig()
    assert cfg.d_search_strategy == "adaptive"
    assert cfg.d_fine_step == 0.01
    assert cfg.d_fine_radius == 0.09
    assert cfg.d_coarse_grid is None


def test_validate_config_rejects_invalid_d_search_strategy():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(d_search_strategy="nope"))


@pytest.mark.parametrize("bad_step", [0.0, -0.01])
def test_validate_d_fine_step_rejects_non_positive(bad_step):
    with pytest.raises(InvalidConfigurationError):
        validate_d_fine_step(bad_step)


def test_validate_d_fine_step_rejects_bool():
    with pytest.raises(InvalidConfigurationError):
        validate_d_fine_step(True)


@pytest.mark.parametrize("bad_radius", [0.0, -0.09])
def test_validate_d_fine_radius_rejects_non_positive(bad_radius):
    with pytest.raises(InvalidConfigurationError):
        validate_d_fine_radius(bad_radius)


def test_validate_d_fine_radius_rejects_bool():
    with pytest.raises(InvalidConfigurationError):
        validate_d_fine_radius(False)


def test_validate_d_coarse_grid_accepts_none():
    assert validate_d_coarse_grid(None) is None


def test_validate_d_coarse_grid_accepts_valid():
    result = validate_d_coarse_grid([0.0, 0.5, 1.0])
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("bad_grid", [[], [0.1, np.nan], [0.1, np.inf], [-0.1, 0.2], [0.2, 1.2], [0.1, "bad"]])
def test_validate_d_coarse_grid_rejects_invalid(bad_grid):
    with pytest.raises(InvalidConfigurationError):
        validate_d_coarse_grid(bad_grid)


def test_validate_config_rejects_bad_d_fine_step():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(d_fine_step=0.0))


def test_validate_config_rejects_bad_d_fine_radius():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(d_fine_radius=-1.0))


def test_validate_config_rejects_bad_d_coarse_grid():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(d_coarse_grid=np.array([1.5])))


# ---------------------------------------------------------------------------
# validate_mode
# ---------------------------------------------------------------------------


def test_validate_config_accepts_valid_modes():
    cfg = CyclicalTestConfig(
        variance_mode="time",
        statistic_mode="test",
        stochastic_cycle_mode="single",
    )
    validated = validate_config(cfg)
    assert validated is cfg


def test_validate_config_rejects_invalid_variance_mode():
    cfg = CyclicalTestConfig(variance_mode="unknown")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_statistic_mode():
    cfg = CyclicalTestConfig(statistic_mode="bad_mode")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_stochastic_cycle_mode():
    cfg = CyclicalTestConfig(stochastic_cycle_mode="triple")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_error_model():
    cfg = CyclicalTestConfig(error_model="ar3")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# validate_boolean
# ---------------------------------------------------------------------------


def test_validate_boolean_accepts_true():
    assert validate_boolean(True, "some_flag") is True


def test_validate_boolean_accepts_false():
    assert validate_boolean(False, "some_flag") is False


def test_validate_config_rejects_non_boolean_flags():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(include_intercept="yes"))  # type: ignore

    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(drop_singular_frequency=1))  # type: ignore

    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(exclude_zero_frequency="false"))  # type: ignore


# ---------------------------------------------------------------------------
# validate_cycle
# ---------------------------------------------------------------------------


def test_validate_cycle_accepts_valid_cycle():
    c = StochasticCycle(R=25, D=0.4)
    assert validate_cycle(c) is c


def test_validate_cycle_accepts_zero_R():
    c = StochasticCycle(R=0, D=0.4)
    assert validate_cycle(c) is c


def test_validate_cycle_rejects_non_integer_R():
    c = StochasticCycle(R=25.5, D=0.4)  # type: ignore
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


def test_validate_cycle_rejects_negative_R():
    c = StochasticCycle(R=-1, D=0.4)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


def test_validate_cycle_rejects_R_greater_or_equal_T_when_T_given():
    c = StochasticCycle(R=100, D=0.4)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c, T=100)


def test_validate_cycle_rejects_negative_D():
    c = StochasticCycle(R=10, D=-0.1)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


def test_validate_cycle_rejects_D_above_one():
    c = StochasticCycle(R=10, D=1.1)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


# ---------------------------------------------------------------------------
# validate_cycles
# ---------------------------------------------------------------------------


def test_validate_cycles_accepts_single_cycle_tuple():
    cycles = (StochasticCycle(R=25, D=0.4),)
    result = validate_cycles(cycles)
    assert isinstance(result, tuple)
    assert len(result) == 1


def test_validate_cycles_rejects_empty_sequence():
    with pytest.raises(InvalidCycleError):
        validate_cycles([])


def test_validate_cycles_rejects_multi_cycle_when_not_allowed():
    cycles = [StochasticCycle(R=25, D=0.4), StochasticCycle(R=30, D=0.2)]
    with pytest.raises(InvalidCycleError):
        validate_cycles(cycles, allow_multi_cycle=False)


def test_validate_cycles_accepts_multi_cycle_when_allowed():
    cycles = [StochasticCycle(R=25, D=0.4), StochasticCycle(R=30, D=0.2)]
    result = validate_cycles(cycles, allow_multi_cycle=True)
    assert len(result) == 2
