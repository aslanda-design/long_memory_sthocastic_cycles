import numpy as np
import pytest

from cyclical_fractional_test.config import CyclicalTestConfig
from cyclical_fractional_test.evaluation import evaluate_candidate, evaluate_r_with_adaptive_d
from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidCycleError
from cyclical_fractional_test.regression import fit_filtered_regression
from cyclical_fractional_test.results import GridCandidateResult, StochasticCycle


def _config(**kwargs):
    return CyclicalTestConfig(stochastic_cycle_mode="single", **kwargs)


def _make_y_X(T=20, p=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T), rng.standard_normal((T, p))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_grid_candidate_result():
    y, X = _make_y_X()
    result = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.3),), _config())
    assert isinstance(result, GridCandidateResult)


def test_key_fields_are_finite():
    y, X = _make_y_X()
    result = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.3),), _config())
    assert np.isfinite(result.test_value)
    assert np.isfinite(result.test_star_value)
    assert np.isfinite(result.xa)
    assert result.xaa > 0.0


def test_R_zero_candidate_runs():
    y, X = _make_y_X()
    result = evaluate_candidate(y, X, (StochasticCycle(R=0, D=0.3),), _config())
    assert result.cycles[0].R == 0
    assert np.isfinite(result.test_value)


def test_output_shapes():
    T, p = 20, 3
    y, X = _make_y_X(T=T, p=p)
    result = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.3),), _config())
    assert result.betas.shape == (p,)
    assert result.beta_standard_errors.shape == (p,)
    assert result.residuals.shape == (T,)


def test_abs_values_consistent_with_signed():
    y, X = _make_y_X()
    result = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.3),), _config())
    assert np.isclose(result.abs_test_value, abs(result.test_value))
    assert np.isclose(result.abs_test_star_value, abs(result.test_star_value))


def test_D_zero_matches_direct_ols():
    """D=0 → filter is identity → betas and residuals match plain OLS."""
    y, X = _make_y_X(T=25, p=2, seed=7)
    result = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.0),), _config())
    direct = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, direct.betas, atol=1e-10)
    np.testing.assert_allclose(
        result.beta_standard_errors, direct.beta_standard_errors, atol=1e-10
    )
    np.testing.assert_allclose(result.residuals, direct.residuals, atol=1e-10)


def test_different_candidates_give_different_results():
    y, X = _make_y_X(T=30, p=2, seed=42)
    r1 = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.0),), _config())
    r2 = evaluate_candidate(y, X, (StochasticCycle(R=4, D=0.5),), _config())
    assert not np.isclose(r1.test_value, r2.test_value)


def test_cycles_field_matches_input():
    y, X = _make_y_X()
    cycles = (StochasticCycle(R=2, D=0.3),)
    assert evaluate_candidate(y, X, cycles, _config()).cycles == cycles


def test_white_noise_metadata_is_default():
    y, X = _make_y_X()
    result = evaluate_candidate(y, X, (StochasticCycle(R=2, D=0.3),), _config())
    assert result.error_model == "white_noise"
    assert result.ar_coefficients == ()


@pytest.mark.parametrize("error_model, coefficient_count", [("ar1", 1), ("ar2", 2)])
def test_ar_error_model_exposes_estimated_coefficients(error_model, coefficient_count):
    y, X = _make_y_X(T=30)
    result = evaluate_candidate(
        y,
        X,
        (StochasticCycle(R=2, D=0.3),),
        _config(error_model=error_model),
    )
    assert result.error_model == error_model
    assert len(result.ar_coefficients) == coefficient_count


def test_adaptive_d_search_accepts_R_zero():
    y, X = _make_y_X(T=30)
    config = _config(
        d_coarse_grid=np.array([0.0, 0.5]),
        d_fine_step=0.25,
        d_fine_radius=0.25,
    )
    result = evaluate_r_with_adaptive_d(y, X, R=0, config=config)
    assert result.R == 0
    assert result.best_result.cycles[0].R == 0
    assert np.isfinite(result.best_result.test_value)


# ---------------------------------------------------------------------------
# Mode dispatch guards
# ---------------------------------------------------------------------------


def test_single_mode_rejects_wrong_cycle_count():
    y, X = _make_y_X()
    with pytest.raises((InvalidCycleError, InvalidConfigurationError, ValueError)):
        evaluate_candidate(y, X, [], _config())
    with pytest.raises((InvalidCycleError, InvalidConfigurationError, ValueError)):
        evaluate_candidate(
            y, X, [StochasticCycle(R=1, D=0.3), StochasticCycle(R=2, D=0.2)], _config()
        )


def test_multi_cycle_mode_runs_with_joint_cycle_tuple():
    y, X = _make_y_X()
    config = CyclicalTestConfig(stochastic_cycle_mode="multi_cycle")
    result = evaluate_candidate(
        y, X, [StochasticCycle(R=1, D=0.3), StochasticCycle(R=2, D=0.2)], config
    )
    assert len(result.cycles) == 2
    assert np.isfinite(result.test_value)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_y, bad_X", [
    (np.ones((20, 1)), np.ones((20, 2))),
    (np.ones(20), np.ones(20)),
    (np.ones(20), np.ones((15, 2))),
])
def test_rejects_invalid_shapes(bad_y, bad_X):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        evaluate_candidate(bad_y, bad_X, [StochasticCycle(R=2, D=0.3)], _config())
