import numpy as np
import pytest

from cyclical_fractional_test.chebyshev import build_chebyshev_design, build_chebyshev_design_at
from cyclical_fractional_test.exceptions import InvalidConfigurationError
from cyclical_fractional_test.prediction import (
    compute_ma_weights,
    forecast_ar,
    forecast_out_of_sample,
    reconstruct_in_sample,
)
from cyclical_fractional_test.results import StochasticCycle


def _filter(x, mu, D):
    """Apply (1 - 2 mu L + L^2)^D to x by truncated MA convolution."""
    from cyclical_fractional_test.filters import compute_fractional_coefficients_from_mu

    coeffs = compute_fractional_coefficients_from_mu(mu, D, len(x))
    return np.convolve(x, coeffs)[: len(x)]


def _fit_pieces(y, R, D, error_model="white_noise"):
    """Build the (X, betas, residuals, ar) tuple a fit would produce for given (R, D)."""
    from cyclical_fractional_test.filters import (
        compute_mu,
        filter_response_and_design,
    )
    from cyclical_fractional_test.regression import estimate_ar_ols, fit_filtered_regression

    T = len(y)
    X = build_chebyshev_design(T, 4, False)
    cycles = (StochasticCycle(R=R, D=D),)
    y_f, X_f = filter_response_and_design(y, X, cycles, mode="single")
    reg = fit_filtered_regression(y_f, X_f)
    order = {"white_noise": 0, "ar1": 1, "ar2": 2}[error_model]
    ar = estimate_ar_ols(reg.residuals, order)
    return X, reg.betas, reg.residuals, ar, cycles


# ---------------------------------------------------------------------------
# reconstruct_in_sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error_model", ["white_noise", "ar1", "ar2"])
def test_in_sample_identity_residual_equals_innovation(error_model):
    rng = np.random.default_rng(3)
    t = np.arange(1, 81)
    y = np.cos(2 * np.pi * 6 * t / 80) + 0.2 * rng.standard_normal(80)
    X, betas, residuals, ar, cycles = _fit_pieces(y, 6, 0.4, error_model)

    yhat = reconstruct_in_sample(y, X, cycles, betas, residuals, ar, "single")

    p = len(ar)
    e = np.array(
        [residuals[a] - sum(ar[i - 1] * residuals[a - i] for i in range(1, p + 1) if a - i >= 0)
         for a in range(len(residuals))]
    )
    np.testing.assert_allclose(y - yhat, e, atol=1e-10)


def test_in_sample_d_zero_is_deterministic_fit():
    # D = 0 is the identity filter, so residuals = y - X beta and the one-step
    # white-noise reconstruction collapses to the deterministic OLS fit X beta.
    rng = np.random.default_rng(5)
    t = np.arange(1, 61)
    y = 0.5 * t / 60 + 0.1 * rng.standard_normal(60)
    X, betas, residuals, ar, cycles = _fit_pieces(y, 5, 0.0, "white_noise")

    yhat = reconstruct_in_sample(y, X, cycles, betas, residuals, ar, "single")
    np.testing.assert_allclose(yhat, X.dot(betas), atol=1e-10)


# ---------------------------------------------------------------------------
# forecast_ar
# ---------------------------------------------------------------------------


def test_forecast_ar_white_noise_is_zero():
    residuals = np.array([1.0, -2.0, 0.5, 0.3, -0.1])
    out = forecast_ar(residuals, np.array([]), 4)
    np.testing.assert_array_equal(out, np.zeros(4))


def test_forecast_ar1_matches_manual_recursion():
    residuals = np.array([0.4, -0.2, 0.6, 0.1, -0.3])
    phi = np.array([0.5])
    out = forecast_ar(residuals, phi, 3)
    expected = np.empty(3)
    last = residuals[-1]
    for k in range(3):
        last = 0.5 * last
        expected[k] = last
    np.testing.assert_allclose(out, expected)


def test_forecast_ar2_matches_manual_recursion():
    residuals = np.array([0.4, -0.2, 0.6, 0.1, -0.3])
    phi = np.array([0.3, 0.2])
    out = forecast_ar(residuals, phi, 3)
    hist = list(residuals)
    expected = []
    for _ in range(3):
        val = 0.3 * hist[-1] + 0.2 * hist[-2]
        hist.append(val)
        expected.append(val)
    np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# forecast_out_of_sample
# ---------------------------------------------------------------------------


def test_forecast_length_and_finiteness():
    rng = np.random.default_rng(7)
    t = np.arange(1, 91)
    y = np.cos(2 * np.pi * 9 * t / 90) + 0.2 * rng.standard_normal(90)
    X, betas, residuals, ar, cycles = _fit_pieces(y, 9, 0.3, "ar1")
    T = len(y)
    X_future = build_chebyshev_design_at(np.arange(T + 1, T + 11), T, 4, False)

    fut = forecast_out_of_sample(y, X, X_future, cycles, betas, residuals, ar, "single", 10)
    assert fut.shape == (10,)
    assert np.all(np.isfinite(fut))


def test_forecast_continuation_recursion_boundary():
    # The first forecast step must use the in-sample S values, matching the
    # deterministic recursion built by hand.
    rng = np.random.default_rng(11)
    t = np.arange(1, 71)
    y = np.cos(2 * np.pi * 7 * t / 70) + 0.2 * rng.standard_normal(70)
    R, D = 7, 0.35
    X, betas, residuals, ar, cycles = _fit_pieces(y, R, D, "white_noise")
    T = len(y)
    from cyclical_fractional_test.filters import compute_fractional_coefficients_from_mu, compute_mu

    mu = compute_mu(T, R)
    C = compute_fractional_coefficients_from_mu(mu, D, T + 1)
    S = y - X.dot(betas)
    # First forecast: eps_{T+1}=0 (white noise) -> S_{T+1} = - sum_{j=1}^{T} C_j S_{T-j+1}
    s_next = -np.dot(C[1 : T + 1], S[::-1])
    X_future = build_chebyshev_design_at(np.array([T + 1.0]), T, 4, False)
    expected_first = X_future.dot(betas)[0] + s_next

    fut = forecast_out_of_sample(y, X, X_future, cycles, betas, residuals, ar, "single", 1)
    np.testing.assert_allclose(fut[0], expected_first, atol=1e-10)


def test_forecast_multi_cycle_boundary_uses_combined_filter():
    rng = np.random.default_rng(12)
    y = rng.standard_normal(48)
    X = np.column_stack([np.ones_like(y), np.linspace(-1.0, 1.0, len(y))])
    betas = np.array([0.2, -0.3])
    residuals = rng.standard_normal(len(y))
    ar = np.array([])
    cycles = (
        StochasticCycle(R=4, D=0.25),
        StochasticCycle(R=9, D=0.15),
    )
    T = len(y)

    from cyclical_fractional_test.filters import (
        compute_fractional_coefficients_from_mu,
        compute_mu,
    )

    combined = np.zeros(T + 1)
    combined[0] = 1.0
    for cycle in cycles:
        coeffs = compute_fractional_coefficients_from_mu(
            compute_mu(T, cycle.R), cycle.D, T + 1
        )
        combined = np.convolve(combined, coeffs, mode="full")[: T + 1]
    S = y - X.dot(betas)
    s_next = -np.dot(combined[1 : T + 1], S[::-1])
    X_future = np.array([[1.0, 1.1]])
    expected_first = X_future.dot(betas)[0] + s_next

    fut = forecast_out_of_sample(
        y,
        X,
        X_future,
        cycles,
        betas,
        residuals,
        ar,
        "multi_cycle",
        1,
    )

    np.testing.assert_allclose(fut[0], expected_first, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_ma_weights
# ---------------------------------------------------------------------------


def test_ma_weights_first_is_one():
    weights = compute_ma_weights((StochasticCycle(R=5, D=0.4),), np.array([]), "single", 80, 6)
    assert np.isclose(weights[0], 1.0)


def test_ma_weights_d_zero_white_noise_is_unit_impulse():
    weights = compute_ma_weights((StochasticCycle(R=5, D=0.0),), np.array([]), "single", 80, 5)
    expected = np.zeros(5)
    expected[0] = 1.0
    np.testing.assert_allclose(weights, expected, atol=1e-12)


def test_ma_weights_multi_cycle_matches_inverse_filter_and_ar_convolution():
    cycles = (
        StochasticCycle(R=5, D=0.3),
        StochasticCycle(R=11, D=0.2),
    )
    phi = np.array([0.4])
    T_ref = 96
    length = 8

    from cyclical_fractional_test.filters import (
        compute_fractional_coefficients_from_mu,
        compute_mu,
    )

    inverse_filter = np.zeros(length)
    inverse_filter[0] = 1.0
    for cycle in cycles:
        coeffs = compute_fractional_coefficients_from_mu(
            compute_mu(T_ref, cycle.R), -cycle.D, length
        )
        inverse_filter = np.convolve(inverse_filter, coeffs, mode="full")[:length]

    ar_weights = np.zeros(length)
    ar_weights[0] = 1.0
    for lag in range(1, length):
        ar_weights[lag] = phi[0] * ar_weights[lag - 1]
    expected = np.convolve(inverse_filter, ar_weights, mode="full")[:length]

    result = compute_ma_weights(cycles, phi, "multi_cycle", T_ref, length)

    np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_reconstruct_rejects_bad_betas_length():
    y = np.ones(20)
    X = build_chebyshev_design(20, 4, False)
    with pytest.raises(InvalidConfigurationError):
        reconstruct_in_sample(y, X, (StochasticCycle(R=3, D=0.2),), np.ones(99), np.ones(20), np.array([]), "single")


def test_forecast_rejects_bad_horizon():
    y = np.ones(20)
    X = build_chebyshev_design(20, 4, False)
    X_future = build_chebyshev_design_at(np.arange(21, 24), 20, 4, False)
    with pytest.raises(InvalidConfigurationError):
        forecast_out_of_sample(y, X, X_future, (StochasticCycle(R=3, D=0.2),), np.ones(4), np.ones(20), np.array([]), "single", 0)
