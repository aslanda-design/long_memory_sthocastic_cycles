import numpy as np
import pytest

from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidCycleError
from cyclical_fractional_test.filters import (
    apply_filter_dynamic,
    apply_fractional_filter_single_series,
    apply_multi_cycle_filter,
    apply_single_cycle_filter,
    compute_fractional_coefficients_dynamic,
    compute_fractional_coefficients_multi_cycle,
    compute_fractional_coefficients_single_cycle,
    compute_mu,
    filter_response_and_design,
)
from cyclical_fractional_test.results import StochasticCycle


# ---------------------------------------------------------------------------
# compute_mu
# ---------------------------------------------------------------------------


def test_compute_mu_matches_formula():
    T, R = 10, 2
    assert np.isclose(compute_mu(T, R), np.cos(2 * np.pi * R / T))


def test_compute_mu_accepts_zero_frequency():
    assert compute_mu(10, 0) == pytest.approx(1.0)


@pytest.mark.parametrize("R", [-1, 10, 11])
def test_compute_mu_rejects_R_out_of_range(R):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_mu(10, R)


@pytest.mark.parametrize("T", [0, 1, -5])
def test_compute_mu_rejects_invalid_T(T):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_mu(T, 1)


def test_compute_mu_rejects_booleans():
    with pytest.raises((InvalidConfigurationError, TypeError)):
        compute_mu(True, 1)
    with pytest.raises((InvalidConfigurationError, TypeError)):
        compute_mu(10, True)


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle
# ---------------------------------------------------------------------------


def test_d_zero_is_identity_filter():
    """(1 - 2μL + L²)⁰ = 1 → C_0=1, C_j=0 for j≥1."""
    coeffs = compute_fractional_coefficients_single_cycle(T=8, R=2, D=0.0)
    expected = np.zeros(8)
    expected[0] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-15)


def test_d_one_matches_second_order_polynomial():
    """(1 - 2μL + L²)¹ → C_0=1, C_1=-2μ, C_2=1, rest=0."""
    T, R = 8, 2
    mu = compute_mu(T, R)
    coeffs = compute_fractional_coefficients_single_cycle(T=T, R=R, D=1.0)
    expected = np.zeros(T)
    expected[0] = 1.0
    expected[1] = -2.0 * mu
    expected[2] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)


def test_r_zero_d_one_matches_non_cyclic_second_difference():
    """R=0, D=1 → (1 - 2L + L²) = (1 - L)²."""
    coeffs = compute_fractional_coefficients_single_cycle(T=8, R=0, D=1.0)
    expected = np.zeros(8)
    expected[0] = 1.0
    expected[1] = -2.0
    expected[2] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)


def test_fractional_d_satisfies_recurrence():
    """Verify C_j = [2μ(j-1-D)C_{j-1} + (2D-j+2)C_{j-2}] / j for j≥2."""
    T, R, D = 5, 1, 0.5
    mu = compute_mu(T, R)
    coeffs = compute_fractional_coefficients_single_cycle(T=T, R=R, D=D)
    for j in range(2, T):
        expected = (2 * mu * (j - 1 - D) * coeffs[j - 1] + (2 * D - j + 2) * coeffs[j - 2]) / j
        assert np.isclose(coeffs[j], expected), f"Recurrence failed at j={j}"


def test_fractional_coefficients_output_shape_and_finiteness():
    coeffs = compute_fractional_coefficients_single_cycle(T=20, R=3, D=0.4)
    assert coeffs.shape == (20,)
    assert np.all(np.isfinite(coeffs))


@pytest.mark.parametrize("bad_D", [float("nan"), float("inf"), True])
def test_fractional_coefficients_rejects_invalid_D(bad_D):
    with pytest.raises((InvalidConfigurationError, ValueError, TypeError)):
        compute_fractional_coefficients_single_cycle(10, 2, bad_D)


@pytest.mark.parametrize("T,R", [(0, 1), (1, 1), (10, -1), (10, 10)])
def test_fractional_coefficients_rejects_invalid_T_R(T, R):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(T, R, 0.5)


def test_fractional_coefficients_dynamic_matches_direct():
    T, R, D = 20, 2, 0.5
    cycles = [StochasticCycle(R=R, D=D)]
    expected = compute_fractional_coefficients_single_cycle(T, R, D)
    result = compute_fractional_coefficients_dynamic(T, cycles, mode="single")
    np.testing.assert_allclose(result, expected)


def test_fractional_coefficients_multi_peak_mode_uses_single_cycle_path():
    T, R, D = 20, 3, 0.4
    cycles = [StochasticCycle(R=R, D=D)]
    expected = compute_fractional_coefficients_single_cycle(T, R, D)

    result = compute_fractional_coefficients_dynamic(
        T, cycles, mode="multi_peak_single_cycle"
    )

    np.testing.assert_allclose(result, expected)


def test_fractional_coefficients_multi_cycle_matches_truncated_convolution():
    T = 12
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=4, D=0.2)]
    c1 = compute_fractional_coefficients_single_cycle(T, cycles[0].R, cycles[0].D)
    c2 = compute_fractional_coefficients_single_cycle(T, cycles[1].R, cycles[1].D)
    expected = np.convolve(c1, c2, mode="full")[:T]
    np.testing.assert_allclose(
        compute_fractional_coefficients_multi_cycle(T, cycles), expected
    )


def test_fractional_coefficients_dynamic_multi_cycle_matches_direct():
    T = 12
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=4, D=0.2)]
    np.testing.assert_allclose(
        compute_fractional_coefficients_dynamic(T, cycles, mode="multi_cycle"),
        compute_fractional_coefficients_multi_cycle(T, cycles),
    )


# ---------------------------------------------------------------------------
# apply_fractional_filter_single_series
# ---------------------------------------------------------------------------


def test_identity_filter_preserves_input():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    coeffs = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(apply_fractional_filter_single_series(x, coeffs), x)


def test_filter_is_causal():
    """out[t] = sum_{j=0}^{t} C_j x[t-j]."""
    x = np.array([1.0, 2.0, 3.0])
    coeffs = np.array([1.0, 10.0, 100.0])
    np.testing.assert_allclose(
        apply_fractional_filter_single_series(x, coeffs), [1.0, 12.0, 123.0]
    )


def test_filter_output_length_matches_input():
    x = np.ones(7)
    assert len(apply_fractional_filter_single_series(x, np.ones(7))) == 7


def test_filter_rejects_coefficients_shorter_than_input():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_fractional_filter_single_series(np.ones(5), np.ones(3))


def test_filter_rejects_2d_input():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_fractional_filter_single_series(np.ones((3, 2)), np.ones(6))


# ---------------------------------------------------------------------------
# apply_single_cycle_filter
# ---------------------------------------------------------------------------


def test_single_cycle_D_zero_is_identity():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    np.testing.assert_allclose(apply_single_cycle_filter(x, StochasticCycle(R=2, D=0.0), T=5), x)


def test_single_cycle_D_one_matches_analytical():
    """(1 - 2μL + L²): out[t] = x[t] - 2μ x[t-1] + x[t-2] for t≥2."""
    T, R = 6, 2
    mu = compute_mu(T, R)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = apply_single_cycle_filter(x, StochasticCycle(R=R, D=1.0), T=T)
    expected = np.zeros(T)
    expected[0] = x[0]
    expected[1] = x[1] - 2.0 * mu * x[0]
    for t in range(2, T):
        expected[t] = x[t] - 2.0 * mu * x[t - 1] + x[t - 2]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_single_cycle_output_shape():
    T = 10
    x = np.random.default_rng(0).standard_normal(T)
    assert apply_single_cycle_filter(x, StochasticCycle(R=3, D=0.4), T=T).shape == (T,)


# ---------------------------------------------------------------------------
# filter_response_and_design
# ---------------------------------------------------------------------------


def test_filter_response_shapes_unchanged():
    T, p = 10, 3
    rng = np.random.default_rng(42)
    y = rng.standard_normal(T)
    X = rng.standard_normal((T, p))
    y_f, X_f = filter_response_and_design(y, X, [StochasticCycle(R=2, D=0.4)])
    assert y_f.shape == (T,)
    assert X_f.shape == (T, p)


def test_filter_response_accepts_design_with_no_columns():
    T = 10
    rng = np.random.default_rng(42)
    y = rng.standard_normal(T)
    X = np.empty((T, 0), dtype=float)
    y_f, X_f = filter_response_and_design(y, X, [StochasticCycle(R=2, D=0.4)])
    assert y_f.shape == (T,)
    assert X_f.shape == (T, 0)


def test_filter_D_zero_leaves_y_and_X_unchanged():
    T = 8
    y = np.random.default_rng(0).standard_normal(T)
    X = np.random.default_rng(1).standard_normal((T, 2))
    y_f, X_f = filter_response_and_design(y, X, [StochasticCycle(R=2, D=0.0)])
    np.testing.assert_allclose(y_f, y)
    np.testing.assert_allclose(X_f, X)


def test_filter_does_not_modify_inputs():
    T = 10
    y = np.random.default_rng(5).standard_normal(T)
    X = np.random.default_rng(6).standard_normal((T, 2))
    y_orig, X_orig = y.copy(), X.copy()
    filter_response_and_design(y, X, [StochasticCycle(R=2, D=0.4)])
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(X, X_orig)


@pytest.mark.parametrize("bad_y, bad_X", [
    (np.ones((5, 1)), np.ones((5, 2))),
    (np.ones(5), np.ones(5)),
    (np.ones(5), np.ones((7, 2))),
])
def test_filter_rejects_invalid_shapes(bad_y, bad_X):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        filter_response_and_design(bad_y, bad_X, [StochasticCycle(R=2, D=0.3)])


def test_apply_filter_dynamic_single_mode_matches_direct():
    T = 10
    x = np.random.default_rng(7).standard_normal(T)
    cycles = [StochasticCycle(R=2, D=0.5)]
    expected = apply_single_cycle_filter(x, cycles[0], T)
    np.testing.assert_allclose(apply_filter_dynamic(x, cycles, T, mode="single"), expected)


def test_apply_filter_dynamic_multi_peak_mode_matches_direct():
    T = 10
    x = np.random.default_rng(10).standard_normal(T)
    cycles = [StochasticCycle(R=2, D=0.5)]
    expected = apply_single_cycle_filter(x, cycles[0], T)

    result = apply_filter_dynamic(
        x, cycles, T, mode="multi_peak_single_cycle"
    )

    np.testing.assert_allclose(result, expected)


def test_multi_cycle_coefficients_match_chained_filter():
    T = 14
    x = np.random.default_rng(8).standard_normal(T)
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=5, D=0.3)]
    coeffs = compute_fractional_coefficients_multi_cycle(T, cycles)
    expected = apply_multi_cycle_filter(x, cycles, T)
    np.testing.assert_allclose(
        apply_fractional_filter_single_series(x, coeffs), expected
    )


def test_apply_filter_dynamic_multi_cycle_matches_direct():
    T = 14
    x = np.random.default_rng(9).standard_normal(T)
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=5, D=0.3)]
    np.testing.assert_allclose(
        apply_filter_dynamic(x, cycles, T, mode="multi_cycle"),
        apply_multi_cycle_filter(x, cycles, T),
    )
