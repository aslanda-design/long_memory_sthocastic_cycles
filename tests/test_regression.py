import numpy as np
import pytest

from cyclical_fractional_test.exceptions import InvalidConfigurationError
from cyclical_fractional_test.regression import (
    BetaSignificanceResult,
    RegressionResult,
    compute_beta_standard_errors,
    compute_beta_t_statistics,
    compute_residual_sum_squares,
    compute_residuals,
    compute_time_variance,
    detect_beta_significance,
    estimate_ar_ols,
    estimate_innovation_variance,
    fit_filtered_regression,
)


# ---------------------------------------------------------------------------
# estimate_innovation_variance
# ---------------------------------------------------------------------------


def test_innovation_variance_white_noise_is_mean_square():
    r = np.array([1.0, -2.0, 3.0, 0.5])
    assert np.isclose(estimate_innovation_variance(r, np.array([])), np.mean(r ** 2))


def test_innovation_variance_ar1_matches_manual():
    r = np.array([0.4, -0.2, 0.6, 0.1, -0.3])
    phi = np.array([0.5])
    e = r[1:] - 0.5 * r[:-1]
    assert np.isclose(estimate_innovation_variance(r, phi), np.mean(e ** 2))


def test_innovation_variance_rejects_too_short_series():
    with pytest.raises(InvalidConfigurationError):
        estimate_innovation_variance(np.array([1.0]), np.array([0.5]))


# ---------------------------------------------------------------------------
# compute_residuals and compute_residual_sum_squares
# ---------------------------------------------------------------------------


def test_compute_residuals_simple():
    y = np.array([1.0, 2.0, 3.0])
    fitted = np.array([1.0, 1.5, 4.0])
    np.testing.assert_allclose(compute_residuals(y, fitted), [0.0, 0.5, -1.0])


def test_compute_residuals_shape_mismatch_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_residuals(np.ones(3), np.ones(4))


def test_compute_residual_sum_squares_known():
    r = np.array([1.0, -2.0, 3.0])
    assert np.isclose(compute_residual_sum_squares(r), 14.0)


def test_compute_residual_sum_squares_zero():
    assert compute_residual_sum_squares(np.zeros(5)) == 0.0


# ---------------------------------------------------------------------------
# fit_filtered_regression
# ---------------------------------------------------------------------------


def test_fit_recovers_simple_beta():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    beta_true = np.array([2.0])
    y = X @ beta_true
    result = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, beta_true, atol=1e-10)
    np.testing.assert_allclose(result.residuals, np.zeros(4), atol=1e-10)


def test_fit_recovers_multiple_betas():
    rng = np.random.default_rng(99)
    T, p = 20, 3
    X = rng.standard_normal((T, p))
    beta_true = np.array([2.0, -1.0, 0.5])
    y = X @ beta_true
    result = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, beta_true, atol=1e-8)


def test_fit_output_shapes():
    T, p = 15, 4
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, p))
    y = rng.standard_normal(T)
    result = fit_filtered_regression(y, X)
    assert result.betas.shape == (p,)
    assert result.beta_standard_errors.shape == (p,)
    assert result.fitted_values.shape == (T,)
    assert result.residuals.shape == (T,)


def test_fit_accepts_no_regressors():
    y = np.array([1.0, -2.0, 3.0, 0.5])
    X = np.empty((len(y), 0), dtype=float)
    result = fit_filtered_regression(y, X)
    assert result.betas.shape == (0,)
    np.testing.assert_allclose(result.fitted_values, np.zeros_like(y))
    np.testing.assert_allclose(result.residuals, y)
    assert result.rank == 0
    assert result.degrees_of_freedom == len(y)
    assert result.beta_standard_errors.shape == (0,)


def test_fit_returns_regression_result():
    result = fit_filtered_regression(np.ones(5), np.ones((5, 1)))
    assert isinstance(result, RegressionResult)


def test_fit_rank_and_condition_number_are_set():
    result = fit_filtered_regression(np.ones(4), np.eye(4))
    assert isinstance(result.rank, int)
    assert isinstance(result.degrees_of_freedom, int)
    assert np.isfinite(result.condition_number)


def test_fit_computes_beta_standard_errors():
    x = np.arange(1.0, 8.0)
    X = np.column_stack([np.ones_like(x), x])
    y = np.array([2.0, 3.2, 3.9, 6.1, 7.9, 9.8, 11.2])

    result = fit_filtered_regression(y, X)

    sigma_squared = result.residual_sum_squares / (len(y) - result.rank)
    expected = np.sqrt(np.diag(sigma_squared * np.linalg.pinv(X.T @ X)))
    np.testing.assert_allclose(result.beta_standard_errors, expected)
    assert result.degrees_of_freedom == len(y) - result.rank


def test_fit_does_not_modify_inputs():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(10)
    X = rng.standard_normal((10, 2))
    y_orig, X_orig = y.copy(), X.copy()
    fit_filtered_regression(y, X)
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(X, X_orig)


@pytest.mark.parametrize("bad_y, bad_X", [
    (np.ones((5, 1)), np.ones((5, 2))),
    (np.ones(5), np.ones(5)),
    (np.ones(5), np.ones((7, 2))),
])
def test_fit_rejects_invalid_shapes(bad_y, bad_X):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(bad_y, bad_X)


def test_fit_rejects_nan_in_inputs():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.array([1.0, float("nan"), 3.0]), np.ones((3, 1)))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.ones(3), np.array([[1.0], [float("nan")], [3.0]]))


# ---------------------------------------------------------------------------
# beta standard errors and significance
# ---------------------------------------------------------------------------


def test_compute_beta_standard_errors_returns_nan_without_degrees_of_freedom():
    X = np.eye(3)
    residuals = np.zeros(3)

    result = compute_beta_standard_errors(X, residuals, rank=3)

    np.testing.assert_array_equal(np.isnan(result), np.ones(3, dtype=bool))


def test_compute_beta_standard_errors_rejects_shape_mismatch():
    with pytest.raises(InvalidConfigurationError):
        compute_beta_standard_errors(np.ones((4, 2)), np.ones(3))


def test_compute_beta_t_statistics_handles_zero_standard_error():
    t_statistics = compute_beta_t_statistics(
        np.array([2.0, 0.0]),
        np.array([0.0, 0.0]),
    )

    assert np.isposinf(t_statistics[0])
    assert np.isnan(t_statistics[1])


def test_detect_beta_significance_uses_absolute_t_values():
    result = detect_beta_significance(
        betas=np.array([2.0, -2.0, 0.5, 0.0]),
        standard_errors=np.array([1.0, 1.0, 1.0, 0.0]),
    )

    assert isinstance(result, BetaSignificanceResult)
    np.testing.assert_allclose(result.t_statistics[:3], [2.0, -2.0, 0.5])
    assert np.isnan(result.t_statistics[3])
    np.testing.assert_array_equal(
        result.significant,
        np.array([True, True, False, False]),
    )
    assert result.critical_value == pytest.approx(1.645)


def test_detect_beta_significance_accepts_custom_critical_value():
    result = detect_beta_significance(
        np.array([1.7]),
        np.array([1.0]),
        critical_value=2.0,
    )

    np.testing.assert_array_equal(result.significant, np.array([False]))


# ---------------------------------------------------------------------------
# compute_time_variance
# ---------------------------------------------------------------------------


def test_time_variance_is_mean_of_squares():
    r = np.array([1.0, -2.0, 3.0])
    assert np.isclose(compute_time_variance(r), 14.0 / 3.0)


def test_time_variance_not_centered():
    r = np.array([1.0, 2.0, 3.0])
    assert np.isclose(compute_time_variance(r), np.mean(r ** 2))
    assert not np.isclose(compute_time_variance(r), np.var(r))


def test_time_variance_zero_residuals():
    assert compute_time_variance(np.zeros(5)) == 0.0


def test_time_variance_rejects_empty_or_2d():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_time_variance(np.array([]))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_time_variance(np.ones((3, 2)))


# ---------------------------------------------------------------------------
# estimate_ar_ols
# ---------------------------------------------------------------------------


def test_estimate_ar_ols_order_zero_returns_empty_array():
    result = estimate_ar_ols(np.array([1.0, 2.0, 3.0]), order=0)
    assert result.shape == (0,)


def test_estimate_ar_ols_recovers_ar1_coefficient():
    residuals = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(estimate_ar_ols(residuals, order=1), [0.5])


def test_estimate_ar_ols_recovers_ar2_coefficients():
    phi_1, phi_2 = 0.5, -0.2
    residuals = [1.0, 0.25]
    for _ in range(12):
        residuals.append(phi_1 * residuals[-1] + phi_2 * residuals[-2])
    np.testing.assert_allclose(
        estimate_ar_ols(np.array(residuals), order=2),
        [phi_1, phi_2],
        atol=1e-12,
    )


@pytest.mark.parametrize("order", [-1, 3, True])
def test_estimate_ar_ols_rejects_unsupported_order(order):
    with pytest.raises(InvalidConfigurationError):
        estimate_ar_ols(np.ones(5), order=order)


def test_estimate_ar_ols_rejects_insufficient_residuals():
    with pytest.raises(InvalidConfigurationError):
        estimate_ar_ols(np.ones(2), order=2)
