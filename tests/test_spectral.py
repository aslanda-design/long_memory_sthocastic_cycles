import numpy as np
import pytest

from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidSeriesError
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.spectral import (
    compute_ar_spectral_adjustment,
    compute_autocorrelogram,
    compute_document_periodogram,
    compute_frequency_variance_dynamic,
    compute_frequency_variance_multi_cycle,
    compute_frequency_variance_single_cycle,
    compute_psi_dynamic,
    compute_psi_multi_cycle,
    compute_psi_single_cycle,
    compute_residual_periodogram,
    compute_xa_ar_adjusted,
    compute_xa_ar1_multi_cycle,
    compute_xa_ar1_single_cycle,
    compute_xa_ar2_multi_cycle,
    compute_xa_ar2_single_cycle,
    compute_xa_dynamic,
    compute_xa_error_model,
    compute_xa_multi_cycle,
    compute_xa_single_cycle,
    compute_xaa_dynamic,
    compute_xaa_ar1_multi_cycle,
    compute_xaa_ar1_single_cycle,
    compute_xaa_ar2_multi_cycle,
    compute_xaa_ar2_single_cycle,
    compute_xaa_error_model,
    compute_xaa_multi_cycle,
    compute_xaa_single_cycle,
    find_periodogram_peak,
    find_top_periodogram_peaks,
)


# ---------------------------------------------------------------------------
# AR error-model adjustments
# ---------------------------------------------------------------------------


def test_ar_spectral_adjustment_white_noise_returns_ones():
    lambdas = 2.0 * np.pi * np.arange(8) / 8
    np.testing.assert_array_equal(
        compute_ar_spectral_adjustment(lambdas, np.array([])),
        np.ones(8),
    )


@pytest.mark.parametrize("coefficients", [np.array([0.4]), np.array([0.5, -0.2])])
def test_ar_spectral_adjustment_is_finite_and_positive(coefficients):
    lambdas = 2.0 * np.pi * np.arange(32) / 32
    adjustment = compute_ar_spectral_adjustment(lambdas, coefficients)
    assert np.all(np.isfinite(adjustment))
    assert np.all(adjustment > 0.0)


def test_ar_spectral_adjustment_protects_near_zero_denominator():
    lambdas = np.array([0.0, np.pi])
    adjustment = compute_ar_spectral_adjustment(lambdas, np.array([1.0]))
    assert np.all(np.isfinite(adjustment))
    assert np.all(adjustment > 0.0)


def test_xa_ar_adjusted_matches_formula():
    psi = np.array([1.0, 2.0, 3.0])
    periodogram = np.array([0.5, 1.0, 1.5])
    adjustment = np.array([1.0, 2.0, 4.0])
    expected = -(2.0 * np.pi / 3.0) * np.sum(psi * periodogram / adjustment)
    assert np.isclose(
        compute_xa_ar_adjusted(psi, periodogram, adjustment), expected
    )


def test_xa_error_model_white_noise_matches_existing_function():
    psi = np.array([1.0, 2.0, 3.0])
    periodogram = np.array([0.5, 1.0, 1.5])
    assert np.isclose(
        compute_xa_error_model(psi, periodogram, "white_noise", np.ones(3)),
        compute_xa_single_cycle(psi, periodogram),
    )


@pytest.mark.parametrize(
    "error_model, coefficients, direct_function",
    [
        ("ar1", np.array([0.4]), compute_xa_ar1_single_cycle),
        ("ar2", np.array([0.5, -0.2]), compute_xa_ar2_single_cycle),
    ],
)
def test_xa_error_model_single_cycle_matches_explicit_ar_function(
    error_model, coefficients, direct_function
):
    lambdas = 2.0 * np.pi * np.arange(4) / 4
    psi = np.array([1.0, 2.0, 3.0, 4.0])
    periodogram = np.array([0.5, 1.0, 1.5, 2.0])
    adjustment = compute_ar_spectral_adjustment(lambdas, coefficients)
    assert np.isclose(
        compute_xa_error_model(
            psi, periodogram, error_model, adjustment, "single"
        ),
        direct_function(psi, periodogram, adjustment),
    )


@pytest.mark.parametrize(
    "error_model, coefficients, direct_function",
    [
        ("ar1", np.array([0.4]), compute_xa_ar1_multi_cycle),
        ("ar2", np.array([0.5, -0.2]), compute_xa_ar2_multi_cycle),
    ],
)
def test_xa_error_model_multi_cycle_matches_explicit_ar_function(
    error_model, coefficients, direct_function
):
    lambdas = 2.0 * np.pi * np.arange(4) / 4
    psi_multi = np.array([1.0, 2.0, 3.0, 4.0])
    periodogram = np.array([0.5, 1.0, 1.5, 2.0])
    adjustment = compute_ar_spectral_adjustment(lambdas, coefficients)
    assert np.isclose(
        compute_xa_error_model(
            psi_multi,
            periodogram,
            error_model,
            adjustment,
            stochastic_cycle_mode="multi_cycle",
        ),
        direct_function(psi_multi, periodogram, adjustment),
    )


@pytest.mark.parametrize(
    "error_model, coefficients, direct_function",
    [
        ("ar1", np.array([0.4]), compute_xaa_ar1_single_cycle),
        ("ar2", np.array([0.5, -0.2]), compute_xaa_ar2_single_cycle),
    ],
)
def test_xaa_error_model_single_cycle_matches_explicit_ar_function(
    error_model, coefficients, direct_function
):
    T = 32
    lambdas = 2.0 * np.pi * np.arange(T) / T
    psi = compute_psi_single_cycle(T, R=4)
    xaa = compute_xaa_error_model(psi, lambdas, error_model, coefficients)
    assert np.isfinite(xaa)
    assert xaa > 0.0
    assert np.isclose(xaa, direct_function(psi, lambdas, coefficients))


def test_xaa_ar1_single_cycle_matches_formula():
    T = 16
    lambdas = 2.0 * np.pi * np.arange(T) / T
    psi = compute_psi_single_cycle(T, R=3)
    coefficients = np.array([0.4])
    adjustment = compute_ar_spectral_adjustment(lambdas, coefficients)
    epsilon = 2.0 * (np.cos(lambdas) - coefficients[0]) * adjustment
    correction = np.sum(psi * epsilon) ** 2 / np.sum(epsilon ** 2)
    expected = (2.0 / T) * (np.sum(psi ** 2) - correction)
    assert np.isclose(
        compute_xaa_ar1_single_cycle(psi, lambdas, coefficients), expected
    )


def test_xaa_ar2_single_cycle_matches_formula():
    T = 16
    lambdas = 2.0 * np.pi * np.arange(T) / T
    psi = compute_psi_single_cycle(T, R=3)
    coefficients = np.array([0.5, -0.2])
    adjustment = compute_ar_spectral_adjustment(lambdas, coefficients)
    phi_1, phi_2 = coefficients
    epsilon = np.column_stack(
        (
            2.0 * (np.cos(lambdas) - phi_1 - phi_2 * np.cos(lambdas))
            * adjustment,
            2.0 * (np.cos(2.0 * lambdas) - phi_1 * np.cos(lambdas) - phi_2)
            * adjustment,
        )
    )
    s_psi_epsilon = epsilon.T @ psi
    correction = s_psi_epsilon.T @ np.linalg.solve(
        epsilon.T @ epsilon, s_psi_epsilon
    )
    expected = (2.0 / T) * (np.sum(psi ** 2) - correction)
    assert np.isclose(
        compute_xaa_ar2_single_cycle(psi, lambdas, coefficients), expected
    )


@pytest.mark.parametrize(
    "error_model, coefficients, direct_function",
    [
        ("ar1", np.array([0.4]), compute_xaa_ar1_multi_cycle),
        ("ar2", np.array([0.5, -0.2]), compute_xaa_ar2_multi_cycle),
    ],
)
def test_xaa_error_model_multi_cycle_matches_explicit_ar_function(
    error_model, coefficients, direct_function
):
    T = 32
    lambdas = 2.0 * np.pi * np.arange(T) / T
    psi_multi = compute_psi_multi_cycle(
        T, [StochasticCycle(R=4, D=0.3), StochasticCycle(R=7, D=0.2)]
    )
    xaa = compute_xaa_error_model(
        psi_multi,
        lambdas,
        error_model,
        coefficients,
        stochastic_cycle_mode="multi_cycle",
    )
    assert np.isfinite(xaa)
    assert xaa > 0.0
    assert np.isclose(xaa, direct_function(psi_multi, lambdas, coefficients))


# ---------------------------------------------------------------------------
# compute_document_periodogram
# ---------------------------------------------------------------------------


def test_periodogram_shapes_and_nonnegativity():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    lambdas, I = compute_document_periodogram(x)
    assert lambdas.shape == (4,)
    assert I.shape == (4,)
    assert np.all(I >= 0)


def test_periodogram_matches_fft_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(x)
    _, I = compute_document_periodogram(x)
    np.testing.assert_allclose(I, np.abs(np.fft.fft(x)) ** 2 / (2.0 * np.pi * T))


def test_periodogram_lambdas_formula():
    T = 8
    lambdas, _ = compute_document_periodogram(np.ones(T))
    np.testing.assert_allclose(lambdas, 2.0 * np.pi * np.arange(T) / T)


def test_periodogram_detects_known_frequency():
    T = 64
    j0 = 5
    x = np.cos(2.0 * np.pi * j0 * np.arange(T, dtype=float) / T)
    _, I = compute_document_periodogram(x)
    peak = find_periodogram_peak(I, exclude_zero=True)
    assert peak == j0 or peak == T - j0


def test_periodogram_rejects_non_finite():
    with pytest.raises(InvalidSeriesError):
        compute_document_periodogram(np.array([1.0, np.nan]))
    with pytest.raises(InvalidSeriesError):
        compute_document_periodogram(np.array([1.0, np.inf, 3.0]))


# ---------------------------------------------------------------------------
# compute_autocorrelogram
# ---------------------------------------------------------------------------


def test_autocorrelogram_matches_manual_unadjusted_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    lags, autocorr = compute_autocorrelogram(x, max_lag=2)
    centered = x - np.mean(x)
    denominator = np.dot(centered, centered)
    expected = np.array([
        1.0,
        np.dot(centered[:-1], centered[1:]) / denominator,
        np.dot(centered[:-2], centered[2:]) / denominator,
    ])
    np.testing.assert_array_equal(lags, np.array([0, 1, 2]))
    np.testing.assert_allclose(autocorr, expected)


def test_autocorrelogram_adjusted_uses_lag_denominator():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    _, unadjusted = compute_autocorrelogram(x, max_lag=2, adjusted=False)
    _, adjusted = compute_autocorrelogram(x, max_lag=2, adjusted=True)
    np.testing.assert_allclose(
        adjusted[1:], unadjusted[1:] * np.array([4 / 3, 4 / 2])
    )


def test_autocorrelogram_default_max_lag_uses_full_series():
    lags, autocorr = compute_autocorrelogram(np.array([1.0, 2.0, 4.0]))
    np.testing.assert_array_equal(lags, np.array([0, 1, 2]))
    assert autocorr.shape == (3,)


def test_autocorrelogram_rejects_bad_inputs():
    with pytest.raises(InvalidConfigurationError):
        compute_autocorrelogram(np.array([1.0, 2.0, 3.0]), max_lag=3)
    with pytest.raises(InvalidConfigurationError):
        compute_autocorrelogram(np.array([1.0, 1.0, 1.0]))
    with pytest.raises(InvalidSeriesError):
        compute_autocorrelogram(np.array([1.0, np.nan, 3.0]))


# ---------------------------------------------------------------------------
# find_periodogram_peak
# ---------------------------------------------------------------------------


def test_find_peak_respects_exclude_zero():
    I = np.array([100.0, 2.0, 5.0, 3.0])
    assert find_periodogram_peak(I, exclude_zero=True) == 2
    assert find_periodogram_peak(I, exclude_zero=False) == 0


def test_find_peak_returns_int():
    assert isinstance(find_periodogram_peak(np.array([0.0, 1.0, 5.0, 2.0])), int)


def test_find_peak_rejects_empty_or_nan():
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([]))
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([1.0, np.nan, 2.0]))


# ---------------------------------------------------------------------------
# find_top_periodogram_peaks
# ---------------------------------------------------------------------------


def test_top_peaks_sorted_descending():
    I = np.array([0.0, 10.0, 5.0, 20.0])
    result = find_top_periodogram_peaks(I, n_peaks=2, exclude_zero=True)
    np.testing.assert_array_equal(result, np.array([3, 1]))


def test_top_peaks_rejects_n_peaks_exceeding_available():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([0.0, 1.0, 2.0]), n_peaks=5, exclude_zero=True)


def test_top_peaks_rejects_non_positive_n_peaks():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([1.0, 2.0, 3.0]), n_peaks=0)


# ---------------------------------------------------------------------------
# compute_residual_periodogram
# ---------------------------------------------------------------------------


def test_residual_periodogram_matches_document_periodogram():
    residuals = np.random.default_rng(0).standard_normal(20)
    lam1, I1 = compute_residual_periodogram(residuals)
    lam2, I2 = compute_document_periodogram(residuals)
    np.testing.assert_array_equal(lam1, lam2)
    np.testing.assert_array_equal(I1, I2)


def test_residual_periodogram_all_zeros_returns_zeros():
    _, I = compute_residual_periodogram(np.zeros(8))
    np.testing.assert_allclose(I, 0.0)


def test_residual_periodogram_output_length_matches_input():
    T = 16
    lambdas, I = compute_residual_periodogram(np.random.default_rng(1).standard_normal(T))
    assert len(lambdas) == T and len(I) == T


# ---------------------------------------------------------------------------
# compute_psi_single_cycle
# ---------------------------------------------------------------------------


def test_psi_shape_and_finiteness():
    psi = compute_psi_single_cycle(T=100, R=25)
    assert psi.shape == (100,)
    assert np.all(np.isfinite(psi))


def test_psi_singular_frequencies_zeroed():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    assert psi[R] == 0.0
    assert psi[T - R] == 0.0


def test_psi_R_equals_T_half_single_zero():
    T, R = 100, 50
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    assert psi[50] == 0.0
    assert np.all(np.isfinite(psi))


def test_psi_R_zero_singular_frequency_zeroed():
    psi = compute_psi_single_cycle(T=100, R=0, drop_singular_frequency=True)
    assert psi[0] == 0.0
    assert np.all(np.isfinite(psi))


def test_psi_matches_formula_at_non_singular_index():
    T, R, j = 100, 25, 10
    psi = compute_psi_single_cycle(T=T, R=R)
    lam_j = 2.0 * np.pi * j / T
    lam_R = 2.0 * np.pi * R / T
    expected = np.log(np.abs(2.0 * (np.cos(lam_j) - np.cos(lam_R))))
    np.testing.assert_allclose(psi[j], expected)


def test_psi_drop_false_gives_neg_inf_at_exact_singular():
    psi = compute_psi_single_cycle(T=100, R=25, drop_singular_frequency=False)
    assert psi[25] == -np.inf


def test_psi_R_zero_drop_false_gives_neg_inf_at_zero():
    psi = compute_psi_single_cycle(T=100, R=0, drop_singular_frequency=False)
    assert psi[0] == -np.inf


@pytest.mark.parametrize("R", [-1, 100])
def test_psi_rejects_invalid_R(R):
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=R)


def test_psi_rejects_non_integer_and_bool_R():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=25.5)  # type: ignore
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=True)  # type: ignore


# ---------------------------------------------------------------------------
# compute_psi_multi_cycle
# ---------------------------------------------------------------------------


def test_psi_multi_cycle_matches_sum_and_zeroes_singular_union():
    T = 40
    cycles = [StochasticCycle(R=4, D=0.3), StochasticCycle(R=9, D=0.2)]
    expected = (
        compute_psi_single_cycle(T, 4, drop_singular_frequency=False)
        + compute_psi_single_cycle(T, 9, drop_singular_frequency=False)
    )
    for idx in [4, T - 4, 9, T - 9]:
        expected[idx] = 0.0
    result = compute_psi_multi_cycle(T, cycles, drop_singular_frequency=True)
    np.testing.assert_allclose(result, expected)
    assert np.all(np.isfinite(result))


def test_psi_multi_cycle_drop_false_keeps_singularities():
    psi = compute_psi_multi_cycle(
        40,
        [StochasticCycle(R=4, D=0.3), StochasticCycle(R=9, D=0.2)],
        drop_singular_frequency=False,
    )
    assert psi[4] == -np.inf
    assert psi[9] == -np.inf


def test_psi_multi_cycle_supports_R_zero():
    T = 40
    cycles = [StochasticCycle(R=0, D=0.3), StochasticCycle(R=9, D=0.2)]
    psi = compute_psi_multi_cycle(T, cycles, drop_singular_frequency=True)
    assert psi[0] == 0.0
    assert psi[9] == 0.0
    assert psi[T - 9] == 0.0
    assert np.all(np.isfinite(psi))


# ---------------------------------------------------------------------------
# compute_xaa_single_cycle
# ---------------------------------------------------------------------------


def test_xaa_matches_formula():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R)
    xaa = compute_xaa_single_cycle(psi)
    np.testing.assert_allclose(xaa, float((2.0 / T) * np.sum(psi ** 2)))


def test_xaa_all_zero_psi_gives_zero():
    assert compute_xaa_single_cycle(np.zeros(50)) == pytest.approx(0.0)


def test_xaa_is_positive():
    psi = compute_psi_single_cycle(T=100, R=25)
    assert compute_xaa_single_cycle(psi) > 0


def test_xaa_rejects_non_finite_or_empty():
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(np.array([]))


# ---------------------------------------------------------------------------
# compute_xaa_multi_cycle
# ---------------------------------------------------------------------------


def test_xaa_multi_cycle_matches_formula():
    T = 40
    psi_multi = compute_psi_multi_cycle(
        T, [StochasticCycle(R=4, D=0.3), StochasticCycle(R=9, D=0.2)]
    )
    expected = (2.0 / T) * np.sum(psi_multi ** 2)
    assert np.isclose(compute_xaa_multi_cycle(psi_multi), expected)


def test_xaa_dynamic_multi_cycle_matches_direct():
    psi_multi = compute_psi_multi_cycle(
        40, [StochasticCycle(R=4, D=0.3), StochasticCycle(R=9, D=0.2)]
    )
    assert np.isclose(
        compute_xaa_dynamic(psi_multi, stochastic_cycle_mode="multi_cycle"),
        compute_xaa_multi_cycle(psi_multi),
    )


# ---------------------------------------------------------------------------
# compute_xa_single_cycle
# ---------------------------------------------------------------------------


def test_xa_known_value():
    psi = np.array([1.0, 2.0, 3.0])
    I = np.array([4.0, 5.0, 6.0])
    expected = -(2.0 * np.pi / 3.0) * (1 * 4 + 2 * 5 + 3 * 6)
    assert np.isclose(compute_xa_single_cycle(psi, I), expected)


def test_xa_sign_is_negative_for_positive_inputs():
    assert compute_xa_single_cycle(np.ones(4), np.ones(4)) < 0.0


def test_xa_zero_inputs_return_zero():
    assert compute_xa_single_cycle(np.zeros(5), np.ones(5)) == 0.0
    assert compute_xa_single_cycle(np.ones(5), np.zeros(5)) == 0.0


def test_xa_rejects_mismatched_shapes():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_xa_single_cycle(np.ones(3), np.ones(4))


def test_xa_rejects_non_finite():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_xa_single_cycle(np.array([1.0, np.nan, 3.0]), np.ones(3))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_xa_single_cycle(np.ones(3), np.array([1.0, np.inf, 3.0]))


# ---------------------------------------------------------------------------
# compute_xa_multi_cycle
# ---------------------------------------------------------------------------


def test_xa_multi_cycle_matches_formula():
    psi_multi = np.array([1.0, 2.0, 3.0, 4.0])
    I = np.array([0.5, 1.0, 1.5, 2.0])
    expected = -(2.0 * np.pi / 4.0) * np.sum(psi_multi * I)
    assert np.isclose(compute_xa_multi_cycle(psi_multi, I), expected)


def test_xa_dynamic_multi_cycle_matches_direct():
    psi_multi = np.array([1.0, 2.0, 3.0, 4.0])
    I = np.array([0.5, 1.0, 1.5, 2.0])
    assert np.isclose(
        compute_xa_dynamic(
            psi_multi,
            I,
            [StochasticCycle(R=1, D=0.3), StochasticCycle(R=2, D=0.2)],
            mode="multi_cycle",
        ),
        compute_xa_multi_cycle(psi_multi, I),
    )


# ---------------------------------------------------------------------------
# compute_frequency_variance_single_cycle
# ---------------------------------------------------------------------------


def test_freq_var_no_drop_sums_all():
    I = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(I)
    expected = (2.0 * np.pi / T) * np.sum(I)
    assert np.isclose(
        compute_frequency_variance_single_cycle(I, R=2, drop_frequency=False), expected
    )


def test_freq_var_drop_excludes_position_R():
    I = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(I)
    expected = (2.0 * np.pi / T) * (1.0 + 2.0 + 4.0)
    assert np.isclose(
        compute_frequency_variance_single_cycle(I, R=2, drop_frequency=True), expected
    )


def test_freq_var_multi_cycle_excludes_all_Rs():
    I = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    T = len(I)
    cycles = [StochasticCycle(R=1, D=0.4), StochasticCycle(R=3, D=0.2)]
    expected = (2.0 * np.pi / T) * (1.0 + 3.0 + 5.0)
    assert np.isclose(
        compute_frequency_variance_multi_cycle(I, cycles, drop_frequency=True), expected
    )


def test_freq_var_rejects_R_out_of_range():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_single_cycle(np.ones(4), R=10)


def test_freq_var_multi_cycle_rejects_empty_cycles():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_multi_cycle(np.ones(4), [])


# ---------------------------------------------------------------------------
# Dispatcher smoke tests
# ---------------------------------------------------------------------------


def _cycle(R=3, D=0.4):
    return StochasticCycle(R=R, D=D)


def test_psi_dynamic_single_matches_direct():
    T, R = 30, 5
    cycles = [_cycle(R=R)]
    expected = compute_psi_single_cycle(T=T, R=R)
    np.testing.assert_allclose(
        compute_psi_dynamic(T, cycles, stochastic_cycle_mode="single"), expected
    )


def test_psi_dynamic_multi_cycle_matches_direct():
    cycles = [_cycle(), _cycle(R=5)]
    np.testing.assert_allclose(
        compute_psi_dynamic(30, cycles, stochastic_cycle_mode="multi_cycle"),
        compute_psi_multi_cycle(30, cycles),
    )


def test_xaa_dynamic_single_matches_direct():
    T, R = 30, 5
    psi = compute_psi_single_cycle(T=T, R=R)
    np.testing.assert_allclose(
        compute_xaa_dynamic(psi, stochastic_cycle_mode="single"),
        compute_xaa_single_cycle(psi),
    )


def test_xa_dynamic_single_matches_direct():
    T = 20
    residuals = np.random.default_rng(1).standard_normal(T)
    psi = compute_psi_single_cycle(T=T, R=3)
    _, I = compute_residual_periodogram(residuals)
    expected = compute_xa_single_cycle(psi, I)
    np.testing.assert_allclose(
        compute_xa_dynamic(psi, I, [_cycle(R=3)], mode="single"), expected
    )


def test_freq_var_dynamic_single_matches_direct():
    I = np.random.default_rng(2).random(20)
    expected = compute_frequency_variance_single_cycle(I, R=3, drop_frequency=True)
    result = compute_frequency_variance_dynamic(I, [_cycle(R=3)], mode="single", drop_frequency=True)
    np.testing.assert_allclose(result, expected)
