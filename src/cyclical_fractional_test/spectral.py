from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .exceptions import InvalidConfigurationError, InvalidCycleError
from .validation import validate_series

_ERROR_MODEL_ORDERS = {"white_noise": 0, "ar1": 1, "ar2": 2}
_STOCHASTIC_CYCLE_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}
_SINGLE_CYCLE_MODES = {"single", "multi_peak_single_cycle"}


# Periodogram and frequency-domain helpers.


def compute_document_periodogram(x: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the periodogram with the normalisation used in the notes.

    I(λ_j) = |FFT(x)_j|² / (2πT),  λ_j = 2πj/T,  j = 0, ..., T-1.

    This matches the sin/cos sum formula; |·|² removes the
    phase shift from 0-based vs 1-based indexing.
    """
    arr = validate_series(x, min_length=2)
    T = len(arr)
    fft_vals = np.fft.fft(arr)
    periodogram = np.abs(fft_vals) ** 2 / (2.0 * np.pi * T)
    lambdas = 2.0 * np.pi * np.arange(T, dtype=float) / T
    return lambdas, periodogram


def compute_autocorrelogram(
    x: Any,
    max_lag: int | None = None,
    adjusted: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sample autocorrelations for lags 0, ..., max_lag.

    The series is demeaned before computing autocovariances. With
    adjusted=False, lag-k autocovariances use the denominator T; with
    adjusted=True, they use T-k. Autocorrelations are normalised by the
    lag-0 autocovariance, so lag 0 is always 1.0.
    """
    arr = validate_series(x, min_length=2)
    T = len(arr)
    lag_limit = T - 1 if max_lag is None else max_lag
    _validate_autocorrelogram_config(lag_limit, T, adjusted)

    centered = arr - np.mean(arr)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise InvalidConfigurationError(
            "autocorrelogram is undefined for a constant series."
        )

    lags = np.arange(lag_limit + 1, dtype=int)
    autocorrelations = np.empty(lag_limit + 1, dtype=float)
    for lag in lags:
        covariance = float(np.dot(centered[: T - lag], centered[lag:]))
        autocorrelations[lag] = covariance / denominator
        if adjusted and lag > 0:
            autocorrelations[lag] *= T / (T - lag)
    autocorrelations[0] = 1.0
    return lags, autocorrelations


def find_periodogram_peak(
    periodogram: np.ndarray,
    exclude_zero: bool = True,
) -> int:
    """Return R* = argmax I(λ_j).

    With exclude_zero=True, frequency 0 is skipped so the mean does not dominate.
    """
    _validate_periodogram(periodogram, min_length=2 if exclude_zero else 1)
    if exclude_zero:
        return int(np.argmax(periodogram[1:])) + 1

    return int(np.argmax(periodogram))


def find_top_periodogram_peaks(
    periodogram: np.ndarray,
    n_peaks: int,
    exclude_zero: bool = True,
) -> np.ndarray:
    """Return the strongest periodogram peaks, largest first."""
    _validate_find_top_peaks(periodogram, n_peaks, exclude_zero)

    candidates = periodogram[1:] if exclude_zero else periodogram
    offset = 1 if exclude_zero else 0
    top_local = np.argsort(candidates)[-n_peaks:][::-1]
    return top_local + offset


def compute_psi_single_cycle(
    T: int,
    R: int,
    drop_singular_frequency: bool = True,
) -> np.ndarray:
    """Compute ψ(λ_j, R) = log(|2(cos(λ_j) - cos(λ_R))|) for j = 0, ..., T-1.

    The expression is singular at j = R and at its mirrored frequency T-R.
    For R=0, the only in-array singularity is j=0.
    When drop_singular_frequency=True, singular positions are set to 0.0.
    """
    _validate_psi_single_cycle(T, R, drop_singular_frequency)
    j = np.arange(T, dtype=float)
    lambda_j = 2.0 * np.pi * j / T
    lambda_R = 2.0 * np.pi * R / T
    with np.errstate(divide="ignore"):
        psi = np.log(np.abs(2.0 * (np.cos(lambda_j) - np.cos(lambda_R))))
    if drop_singular_frequency:
        psi[R] = 0.0
        mirror = T - R
        if 0 <= mirror < T and mirror != R:
            psi[mirror] = 0.0
    return psi


def compute_psi_multi_cycle(
    T: int,
    cycles: object,
    drop_singular_frequency: bool = True,
) -> np.ndarray:
    """Compute ψ_multi(λ_j) = Σ_q log(|2(cos(λ_j) - cos(λ_Rq))|).

    When drop_singular_frequency=True, ψ_multi[j] is set to 0.0 at every
    in-array singular index j in {R_q, T-R_q}; for R_q=0 this is only j=0.
    """
    try:
        cycle_list = tuple(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    _validate_psi_multi_cycle(T, cycle_list, drop_singular_frequency)
    psi_multi = np.zeros(T, dtype=float)
    singular_indices: set[int] = set()
    for cycle in cycle_list:
        psi_multi += compute_psi_single_cycle(
            T, cycle.R, drop_singular_frequency=False
        )
        singular_indices.add(cycle.R)
        mirror = T - cycle.R
        if 0 <= mirror < T:
            singular_indices.add(mirror)
    if drop_singular_frequency:
        for idx in singular_indices:
            psi_multi[idx] = 0.0
    return psi_multi


def compute_xaa_single_cycle(psi: np.ndarray) -> float:
    """Compute XAA(R) = (2/T) * Σ_{j=0}^{T-1} ψ(λ_j, R)².  T = len(psi)."""
    _validate_xaa_single_cycle(psi)
    T = len(psi)
    return float((2.0 / T) * np.sum(psi ** 2))


def compute_xaa_multi_cycle(psi_multi: np.ndarray) -> float:
    """Compute XAA_multi = (2/T) * Σ_j ψ_multi(λ_j)^2.  T = len(psi_multi)."""
    _validate_xaa_single_cycle(psi_multi)
    T = len(psi_multi)
    return float((2.0 / T) * np.sum(psi_multi ** 2))


def compute_ar_spectral_adjustment(
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
) -> np.ndarray:
    """Compute g(lambda) for white-noise, AR(1), or AR(2) residual errors."""
    _validate_compute_ar_spectral_adjustment(lambdas, ar_coefficients)
    lambdas_arr = np.asarray(lambdas, dtype=float)
    coefficients_arr = np.asarray(ar_coefficients, dtype=float)
    if len(coefficients_arr) == 0:
        return np.ones_like(lambdas_arr)

    harmonics = np.arange(1, len(coefficients_arr) + 1, dtype=float)
    ar_polynomial = 1.0 - np.sum(
        coefficients_arr[:, np.newaxis]
        * np.exp(1j * harmonics[:, np.newaxis] * lambdas_arr),
        axis=0,
    )
    denominator = np.abs(ar_polynomial) ** 2
    denominator = np.clip(
        denominator, np.finfo(float).eps, np.finfo(float).max
    )
    adjustment = 1.0 / denominator
    if not np.all(np.isfinite(adjustment)):
        raise InvalidConfigurationError(
            "AR spectral adjustment contains non-finite values."
        )
    return adjustment


def compute_xaa_error_model(
    psi: np.ndarray,
    lambdas: np.ndarray,
    error_model: str,
    ar_coefficients: np.ndarray,
    stochastic_cycle_mode: str = "single",
) -> float:
    """Dispatch XAA by residual error model and stochastic-cycle mode."""
    _validate_compute_xaa_error_model(
        error_model, ar_coefficients, stochastic_cycle_mode
    )
    if error_model == "white_noise":
        return compute_xaa_dynamic(psi, stochastic_cycle_mode)
    if error_model == "ar1":
        return compute_xaa_ar1_dynamic(
            psi, lambdas, ar_coefficients, stochastic_cycle_mode
        )
    return compute_xaa_ar2_dynamic(
        psi, lambdas, ar_coefficients, stochastic_cycle_mode
    )


def compute_xaa_ar1_single_cycle(
    psi: np.ndarray,
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
) -> float:
    """Compute XAA_AR1 = (2/T)[Σψ² - (Σψ epsilon)² / Σεpsilon²]."""
    _validate_compute_xaa_ar1_single_cycle(psi, lambdas, ar_coefficients)
    psi_arr = np.asarray(psi, dtype=float)
    lambdas_arr = np.asarray(lambdas, dtype=float)
    coefficients_arr = np.asarray(ar_coefficients, dtype=float)
    adjustment = compute_ar_spectral_adjustment(lambdas_arr, coefficients_arr)
    epsilon = 2.0 * (np.cos(lambdas_arr) - coefficients_arr[0]) * adjustment
    epsilon_sum_squares = np.sum(epsilon ** 2)
    if epsilon_sum_squares <= np.finfo(float).eps:
        raise InvalidConfigurationError(
            "AR(1) XAA adjustment has a near-zero epsilon denominator."
        )
    correction = np.sum(psi_arr * epsilon) ** 2 / epsilon_sum_squares
    return _compute_ar_adjusted_xaa(psi_arr, correction)


def compute_xaa_ar1_multi_cycle(
    psi_multi: np.ndarray,
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
) -> float:
    """Compute XAA_AR1,multi = (2/T)[Σψ_multi² - (Σψ_multi epsilon)² / Σεpsilon²]."""
    return compute_xaa_ar1_single_cycle(psi_multi, lambdas, ar_coefficients)


def compute_xaa_ar1_dynamic(
    psi: np.ndarray,
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
    stochastic_cycle_mode: str = "single",
) -> float:
    """Dispatch AR(1)-adjusted XAA by stochastic-cycle mode."""
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES:
        return compute_xaa_ar1_single_cycle(psi, lambdas, ar_coefficients)
    if stochastic_cycle_mode == "multi_cycle":
        return compute_xaa_ar1_multi_cycle(psi, lambdas, ar_coefficients)
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
    )


def compute_xaa_ar2_single_cycle(
    psi: np.ndarray,
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
) -> float:
    """Compute XAA_AR2 = (2/T)[Σψ² - S_ψε' inv(S_εε) S_ψε]."""
    _validate_compute_xaa_ar2_single_cycle(psi, lambdas, ar_coefficients)
    psi_arr = np.asarray(psi, dtype=float)
    lambdas_arr = np.asarray(lambdas, dtype=float)
    coefficients_arr = np.asarray(ar_coefficients, dtype=float)
    adjustment = compute_ar_spectral_adjustment(lambdas_arr, coefficients_arr)
    phi_1, phi_2 = coefficients_arr
    epsilon = np.column_stack(
        (
            2.0
            * (
                np.cos(lambdas_arr)
                - phi_1
                - phi_2 * np.cos(lambdas_arr)
            )
            * adjustment,
            2.0
            * (
                np.cos(2.0 * lambdas_arr)
                - phi_1 * np.cos(lambdas_arr)
                - phi_2
            )
            * adjustment,
        )
    )
    s_psi_epsilon = epsilon.T @ psi_arr
    s_epsilon_epsilon = epsilon.T @ epsilon
    try:
        correction = s_psi_epsilon.T @ np.linalg.solve(
            s_epsilon_epsilon, s_psi_epsilon
        )
    except np.linalg.LinAlgError as exc:
        raise InvalidConfigurationError(
            "AR(2) XAA adjustment is singular for the estimated coefficients."
        ) from exc
    return _compute_ar_adjusted_xaa(psi_arr, correction)


def compute_xaa_ar2_multi_cycle(
    psi_multi: np.ndarray,
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
) -> float:
    """Compute XAA_AR2,multi = (2/T)[Σψ_multi² - S_ψε' inv(S_εε) S_ψε]."""
    return compute_xaa_ar2_single_cycle(psi_multi, lambdas, ar_coefficients)


def compute_xaa_ar2_dynamic(
    psi: np.ndarray,
    lambdas: np.ndarray,
    ar_coefficients: np.ndarray,
    stochastic_cycle_mode: str = "single",
) -> float:
    """Dispatch AR(2)-adjusted XAA by stochastic-cycle mode."""
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES:
        return compute_xaa_ar2_single_cycle(psi, lambdas, ar_coefficients)
    if stochastic_cycle_mode == "multi_cycle":
        return compute_xaa_ar2_multi_cycle(psi, lambdas, ar_coefficients)
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
    )


def _compute_ar_adjusted_xaa(psi: np.ndarray, correction: float) -> float:
    T = len(psi)
    xaa = float((2.0 / T) * (np.sum(psi ** 2) - correction))
    if not np.isfinite(xaa) or xaa <= 0.0:
        raise InvalidConfigurationError(
            "AR-adjusted XAA must be finite and positive."
        )
    return xaa


def compute_residual_periodogram(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the periodogram of the regression residuals.

    Delegates to compute_document_periodogram; same normalisation I(λ_j) = |FFT|²/(2πT).
    """
    _validate_compute_residual_periodogram(residuals)
    return compute_document_periodogram(residuals)


def compute_frequency_variance_single_cycle(
    I_residuals: np.ndarray,
    R: int,
    drop_frequency: bool = True,
) -> float:
    """Compute VAR*(R,D) = (2π/T) Σ_j I_residuals[j], optionally excluding j = R."""
    _validate_compute_frequency_variance_single_cycle(I_residuals, R)
    T = len(I_residuals)
    mask = np.ones(T, dtype=bool)
    if drop_frequency:
        mask[R] = False
    return float((2.0 * np.pi / T) * np.sum(I_residuals[mask]))


def compute_frequency_variance_multi_cycle(
    I_residuals: np.ndarray,
    cycles: object,
    drop_frequency: bool = True,
) -> float:
    """Compute VAR*(R,D) = (2π/T) Σ_j I_residuals[j], excluding j = R_q for all cycles."""
    _validate_compute_frequency_variance_multi_cycle(I_residuals, cycles)
    T = len(I_residuals)
    mask = np.ones(T, dtype=bool)
    if drop_frequency:
        for cycle in cycles:
            if 0 <= cycle.R < T:
                mask[cycle.R] = False
    return float((2.0 * np.pi / T) * np.sum(I_residuals[mask]))


def compute_frequency_variance_dynamic(
    I_residuals: np.ndarray,
    cycles: tuple,
    mode: str = "single",
    drop_frequency: bool = True,
) -> float:
    """Dispatch to the single- or multi-cycle frequency variance based on mode.

    "multi_peak_single_cycle" uses the single-cycle path; peak selection already happened upstream.
    """
    _validate_compute_frequency_variance_dynamic(cycles, mode)
    if mode in ("single", "multi_peak_single_cycle"):
        return compute_frequency_variance_single_cycle(
            I_residuals, cycles[0].R, drop_frequency
        )
    return compute_frequency_variance_multi_cycle(I_residuals, cycles, drop_frequency)


def compute_xa_single_cycle(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
) -> float:
    """Compute XA(R,D) = -(2π/T) * Σ_j ψ(λ_j, R) * I_residuals(λ_j).  T = len(psi)."""
    _validate_compute_xa_single_cycle(psi, residual_periodogram)
    T = len(psi)
    return float(-(2.0 * np.pi / T) * np.sum(psi * residual_periodogram))


def compute_xa_multi_cycle(
    psi_multi: np.ndarray,
    residual_periodogram: np.ndarray,
) -> float:
    """Compute XA_multi = -(2π/T) * Σ_j ψ_multi(λ_j) * I_residuals(λ_j)."""
    _validate_compute_xa_single_cycle(psi_multi, residual_periodogram)
    T = len(psi_multi)
    return float(-(2.0 * np.pi / T) * np.sum(psi_multi * residual_periodogram))


def compute_xa_ar_adjusted(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
) -> float:
    """Compute XA_AR = -(2π/T) * Σ_j ψ(λ_j) I_residuals(λ_j) / g(λ_j)."""
    _validate_compute_xa_ar_adjusted(
        psi, residual_periodogram, ar_spectral_adjustment
    )
    return _compute_xa_ar_adjusted(
        psi, residual_periodogram, ar_spectral_adjustment
    )


def _compute_xa_ar_adjusted(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
) -> float:
    T = len(psi)
    return float(
        -(2.0 * np.pi / T)
        * np.sum(psi * residual_periodogram / ar_spectral_adjustment)
    )


def compute_xa_error_model(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    error_model: str,
    ar_spectral_adjustment: np.ndarray,
    stochastic_cycle_mode: str = "single",
) -> float:
    """Dispatch XA by residual error model and stochastic-cycle mode."""
    _validate_compute_xa_error_model(error_model, stochastic_cycle_mode)
    if error_model == "white_noise":
        return compute_xa_dynamic(
            psi, residual_periodogram, mode=stochastic_cycle_mode
        )
    if error_model == "ar1":
        return compute_xa_ar1_dynamic(
            psi,
            residual_periodogram,
            ar_spectral_adjustment,
            stochastic_cycle_mode,
        )
    return compute_xa_ar2_dynamic(
        psi,
        residual_periodogram,
        ar_spectral_adjustment,
        stochastic_cycle_mode,
    )


def compute_xa_ar1_single_cycle(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
) -> float:
    """Compute XA_AR1 = -(2π/T) * Σ_j ψ(λ_j) I_residuals(λ_j) / g_AR1(λ_j)."""
    _validate_compute_xa_ar1_single_cycle(
        psi, residual_periodogram, ar_spectral_adjustment
    )
    return _compute_xa_ar_adjusted(
        psi, residual_periodogram, ar_spectral_adjustment
    )


def compute_xa_ar1_multi_cycle(
    psi_multi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
) -> float:
    """Compute XA_AR1,multi = -(2π/T) * Σ_j ψ_multi(λ_j) I_residuals(λ_j) / g_AR1(λ_j)."""
    return compute_xa_ar1_single_cycle(
        psi_multi, residual_periodogram, ar_spectral_adjustment
    )


def compute_xa_ar1_dynamic(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
    stochastic_cycle_mode: str = "single",
) -> float:
    """Dispatch AR(1)-adjusted XA by stochastic-cycle mode."""
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES:
        return compute_xa_ar1_single_cycle(
            psi, residual_periodogram, ar_spectral_adjustment
        )
    if stochastic_cycle_mode == "multi_cycle":
        return compute_xa_ar1_multi_cycle(
            psi, residual_periodogram, ar_spectral_adjustment
        )
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
    )


def compute_xa_ar2_single_cycle(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
) -> float:
    """Compute XA_AR2 = -(2π/T) * Σ_j ψ(λ_j) I_residuals(λ_j) / g_AR2(λ_j)."""
    _validate_compute_xa_ar2_single_cycle(
        psi, residual_periodogram, ar_spectral_adjustment
    )
    return _compute_xa_ar_adjusted(
        psi, residual_periodogram, ar_spectral_adjustment
    )


def compute_xa_ar2_multi_cycle(
    psi_multi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
) -> float:
    """Compute XA_AR2,multi = -(2π/T) * Σ_j ψ_multi(λ_j) I_residuals(λ_j) / g_AR2(λ_j)."""
    return compute_xa_ar2_single_cycle(
        psi_multi, residual_periodogram, ar_spectral_adjustment
    )


def compute_xa_ar2_dynamic(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    ar_spectral_adjustment: np.ndarray,
    stochastic_cycle_mode: str = "single",
) -> float:
    """Dispatch AR(2)-adjusted XA by stochastic-cycle mode."""
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES:
        return compute_xa_ar2_single_cycle(
            psi, residual_periodogram, ar_spectral_adjustment
        )
    if stochastic_cycle_mode == "multi_cycle":
        return compute_xa_ar2_multi_cycle(
            psi, residual_periodogram, ar_spectral_adjustment
        )
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
    )


def compute_xa_dynamic(
    psi: np.ndarray,
    residual_periodogram: np.ndarray,
    cycles: object = None,
    mode: str = "single",
) -> float:
    """Dispatch to the single- or multi-cycle XA based on mode.

    "multi_peak_single_cycle" uses the single-cycle path; peak selection already happened upstream.
    """
    if mode not in _STOCHASTIC_CYCLE_MODES:
        raise InvalidConfigurationError(
            f"Unknown mode: {mode!r}. "
            f"Expected one of {sorted(_STOCHASTIC_CYCLE_MODES)}."
        )
    if mode in _SINGLE_CYCLE_MODES:
        return compute_xa_single_cycle(psi, residual_periodogram)
    return compute_xa_multi_cycle(psi, residual_periodogram)


def compute_psi_dynamic(
    T: int,
    cycles: tuple,
    stochastic_cycle_mode: str,
    drop_singular_frequency: bool = True,
) -> np.ndarray:
    """Choose the ψ calculation for the selected cycle mode.

    "multi_peak_single_cycle" shares the single-cycle path here because the
    peak selection has already happened upstream.
    """
    _validate_psi_dynamic(cycles, stochastic_cycle_mode)
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES:
        return compute_psi_single_cycle(T, cycles[0].R, drop_singular_frequency)
    return compute_psi_multi_cycle(T, cycles, drop_singular_frequency)


def compute_xaa_dynamic(
    psi: np.ndarray,
    stochastic_cycle_mode: str,
) -> float:
    """Choose the XAA calculation for the selected cycle mode.

    "multi_peak_single_cycle" uses the same single-cycle path as compute_psi_dynamic.
    """
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES:
        return compute_xaa_single_cycle(psi)
    if stochastic_cycle_mode == "multi_cycle":
        return compute_xaa_multi_cycle(psi)
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
    )

# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact 
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_periodogram(periodogram: Any, min_length: int = 1) -> None:
    if not isinstance(periodogram, np.ndarray):
        try:
            periodogram = np.asarray(periodogram, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"periodogram must be a numeric array: {exc}"
            ) from exc
    if periodogram.ndim != 1:
        raise InvalidConfigurationError(
            f"periodogram must be 1-dimensional, got shape {periodogram.shape}."
        )
    if periodogram.size == 0:
        raise InvalidConfigurationError("periodogram must not be empty.")
    if periodogram.size < min_length:
        raise InvalidConfigurationError(
            f"periodogram has {periodogram.size} elements; at least {min_length} required."
        )
    if not np.all(np.isfinite(periodogram)):
        raise InvalidConfigurationError(
            "periodogram contains non-finite values (NaN or inf)."
        )


def _validate_find_top_peaks(
    periodogram: Any, n_peaks: int, exclude_zero: bool
) -> None:
    if isinstance(n_peaks, bool) or not isinstance(n_peaks, int):
        raise InvalidConfigurationError(
            f"n_peaks must be an int, got {type(n_peaks).__name__}."
        )
    if n_peaks < 1:
        raise InvalidConfigurationError(f"n_peaks must be >= 1, got {n_peaks}.")
    _validate_periodogram(periodogram, min_length=1)
    n_available = len(periodogram) - (1 if exclude_zero else 0)
    if n_peaks > n_available:
        raise InvalidConfigurationError(
            f"n_peaks={n_peaks} exceeds the number of available frequencies "
            f"({n_available})."
        )


def _validate_autocorrelogram_config(
    max_lag: int, series_length: int, adjusted: bool
) -> None:
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise InvalidConfigurationError(
            f"max_lag must be an int, got {type(max_lag).__name__}."
        )
    if max_lag < 0:
        raise InvalidConfigurationError(f"max_lag must be >= 0, got {max_lag}.")
    if max_lag >= series_length:
        raise InvalidConfigurationError(
            f"max_lag must be smaller than series length {series_length}, got {max_lag}."
        )
    if not isinstance(adjusted, bool):
        raise InvalidConfigurationError(
            f"adjusted must be a bool, got {type(adjusted).__name__}."
        )


def _validate_psi_single_cycle(T: int, R: int, drop_singular_frequency: bool) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    if isinstance(R, bool) or not isinstance(R, int):
        raise InvalidConfigurationError(f"R must be an int, got {type(R).__name__}.")
    if R < 0 or R > T - 1:
        raise InvalidConfigurationError(
            f"R must satisfy 0 <= R <= T-1={T - 1}, got R={R}."
        )
    if not isinstance(drop_singular_frequency, bool):
        raise InvalidConfigurationError(
            f"drop_singular_frequency must be a bool, "
            f"got {type(drop_singular_frequency).__name__}."
        )


def _validate_psi_multi_cycle(
    T: int, cycles: object, drop_singular_frequency: bool
) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    if not isinstance(drop_singular_frequency, bool):
        raise InvalidConfigurationError(
            f"drop_singular_frequency must be a bool, "
            f"got {type(drop_singular_frequency).__name__}."
        )
    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidCycleError("cycles must not be empty.")
    for cycle in cycle_list:
        _validate_psi_single_cycle(T, cycle.R, drop_singular_frequency)


def _validate_xaa_single_cycle(psi: Any) -> None:
    if not isinstance(psi, np.ndarray):
        try:
            psi = np.asarray(psi, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"psi must be a numeric array: {exc}"
            ) from exc
    if psi.ndim != 1:
        raise InvalidConfigurationError(
            f"psi must be 1-dimensional, got shape {psi.shape}."
        )
    if psi.size == 0:
        raise InvalidConfigurationError("psi must not be empty.")
    if not np.all(np.isfinite(psi)):
        raise InvalidConfigurationError(
            "psi contains non-finite values. "
            "Ensure compute_psi_single_cycle was called with "
            "drop_singular_frequency=True."
        )


def _validate_compute_ar_spectral_adjustment(
    lambdas: Any, ar_coefficients: Any
) -> None:
    _validate_numeric_1d_array(lambdas, "lambdas")
    _validate_ar_coefficients(ar_coefficients)


def _validate_compute_xaa_error_model(
    error_model: str,
    ar_coefficients: Any,
    stochastic_cycle_mode: str,
) -> None:
    _validate_error_model_coefficients(error_model, ar_coefficients)
    _validate_stochastic_cycle_mode(stochastic_cycle_mode)


def _validate_compute_xaa_ar1_single_cycle(
    psi: Any,
    lambdas: Any,
    ar_coefficients: Any,
) -> None:
    _validate_psi_and_lambdas(psi, lambdas)
    _validate_error_model_coefficients("ar1", ar_coefficients)


def _validate_compute_xaa_ar2_single_cycle(
    psi: Any,
    lambdas: Any,
    ar_coefficients: Any,
) -> None:
    _validate_psi_and_lambdas(psi, lambdas)
    _validate_error_model_coefficients("ar2", ar_coefficients)


def _validate_compute_residual_periodogram(residuals: Any) -> None:
    if not isinstance(residuals, np.ndarray):
        try:
            residuals = np.asarray(residuals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"residuals must be a numeric array: {exc}"
            ) from exc
    if residuals.ndim != 1 or residuals.size == 0:
        raise InvalidConfigurationError("residuals must be a non-empty 1-D array.")
    if not np.all(np.isfinite(residuals)):
        raise InvalidConfigurationError("residuals contains non-finite values.")


def _validate_compute_frequency_variance_single_cycle(
    I_residuals: Any, R: int
) -> None:
    if not isinstance(I_residuals, np.ndarray):
        try:
            I_residuals = np.asarray(I_residuals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"I_residuals must be a numeric array: {exc}"
            ) from exc
    if I_residuals.ndim != 1 or I_residuals.size == 0:
        raise InvalidConfigurationError("I_residuals must be a non-empty 1-D array.")
    T = len(I_residuals)
    if isinstance(R, bool) or not isinstance(R, int):
        raise InvalidConfigurationError(f"R must be an int, got {type(R).__name__}.")
    if R < 0 or R >= T:
        raise InvalidConfigurationError(f"R must satisfy 0 <= R < T={T}, got R={R}.")


def _validate_compute_frequency_variance_multi_cycle(
    I_residuals: Any, cycles: object
) -> None:
    if not isinstance(I_residuals, np.ndarray):
        try:
            I_residuals = np.asarray(I_residuals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"I_residuals must be a numeric array: {exc}"
            ) from exc
    if I_residuals.ndim != 1 or I_residuals.size == 0:
        raise InvalidConfigurationError("I_residuals must be a non-empty 1-D array.")
    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")


def _validate_compute_frequency_variance_dynamic(
    cycles: tuple, mode: str
) -> None:
    if mode not in _STOCHASTIC_CYCLE_MODES:
        raise InvalidConfigurationError(
            f"Unknown mode: {mode!r}. "
            f"Expected one of {sorted(_STOCHASTIC_CYCLE_MODES)}."
        )
    if mode in _SINGLE_CYCLE_MODES and len(cycles) != 1:
        raise InvalidCycleError(
            f"mode={mode!r} requires exactly 1 cycle, got {len(cycles)}."
        )


def _validate_compute_xa_single_cycle(
    psi: Any, residual_periodogram: Any
) -> None:
    try:
        psi_arr = np.asarray(psi, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"psi must be a numeric array: {exc}") from exc
    try:
        i_arr = np.asarray(residual_periodogram, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"residual_periodogram must be a numeric array: {exc}"
        ) from exc
    if psi_arr.ndim != 1 or psi_arr.size == 0:
        raise InvalidConfigurationError("psi must be a non-empty 1-D array.")
    if i_arr.ndim != 1 or i_arr.size == 0:
        raise InvalidConfigurationError(
            "residual_periodogram must be a non-empty 1-D array."
        )
    if psi_arr.shape != i_arr.shape:
        raise InvalidConfigurationError(
            f"psi and residual_periodogram must have the same shape; "
            f"got {psi_arr.shape} and {i_arr.shape}."
        )
    if not np.all(np.isfinite(psi_arr)):
        raise InvalidConfigurationError("psi contains non-finite values.")
    if not np.all(np.isfinite(i_arr)):
        raise InvalidConfigurationError(
            "residual_periodogram contains non-finite values."
        )


def _validate_compute_xa_ar_adjusted(
    psi: Any,
    residual_periodogram: Any,
    ar_spectral_adjustment: Any,
) -> None:
    _validate_compute_xa_single_cycle(psi, residual_periodogram)
    _validate_ar_spectral_adjustment_values(
        ar_spectral_adjustment, expected_shape=np.asarray(psi).shape
    )


def _validate_compute_xa_error_model(
    error_model: str,
    stochastic_cycle_mode: str,
) -> None:
    _validate_error_model(error_model)
    _validate_stochastic_cycle_mode(stochastic_cycle_mode)


def _validate_compute_xa_ar1_single_cycle(
    psi: Any,
    residual_periodogram: Any,
    ar_spectral_adjustment: Any,
) -> None:
    _validate_compute_xa_ar_adjusted(
        psi, residual_periodogram, ar_spectral_adjustment
    )


def _validate_compute_xa_ar2_single_cycle(
    psi: Any,
    residual_periodogram: Any,
    ar_spectral_adjustment: Any,
) -> None:
    _validate_compute_xa_ar_adjusted(
        psi, residual_periodogram, ar_spectral_adjustment
    )


def _validate_psi_dynamic(cycles: tuple, stochastic_cycle_mode: str) -> None:
    if stochastic_cycle_mode not in _STOCHASTIC_CYCLE_MODES:
        raise InvalidConfigurationError(
            f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}."
        )
    if stochastic_cycle_mode in _SINGLE_CYCLE_MODES and len(cycles) != 1:
        raise InvalidCycleError(
            f"stochastic_cycle_mode={stochastic_cycle_mode!r} requires exactly "
            f"1 cycle, got {len(cycles)}."
        )


def _validate_numeric_1d_array(values: Any, name: str) -> None:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"{name} must be a numeric array: {exc}") from exc
    if arr.ndim != 1 or arr.size == 0:
        raise InvalidConfigurationError(f"{name} must be a non-empty 1-D array.")
    if not np.all(np.isfinite(arr)):
        raise InvalidConfigurationError(f"{name} contains non-finite values.")


def _validate_psi_and_lambdas(psi: Any, lambdas: Any) -> None:
    _validate_numeric_1d_array(psi, "psi")
    _validate_numeric_1d_array(lambdas, "lambdas")
    if np.asarray(psi).shape != np.asarray(lambdas).shape:
        raise InvalidConfigurationError(
            f"psi and lambdas must have the same shape; "
            f"got {np.asarray(psi).shape} and {np.asarray(lambdas).shape}."
        )


def _validate_ar_coefficients(ar_coefficients: Any) -> None:
    try:
        coefficients = np.asarray(ar_coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"ar_coefficients must be a numeric array: {exc}"
        ) from exc
    if coefficients.ndim != 1:
        raise InvalidConfigurationError(
            f"ar_coefficients must be 1-dimensional, got shape {coefficients.shape}."
        )
    if len(coefficients) > 2:
        raise InvalidConfigurationError(
            f"At most 2 AR coefficients are supported, got {len(coefficients)}."
        )
    if not np.all(np.isfinite(coefficients)):
        raise InvalidConfigurationError(
            "ar_coefficients contains non-finite values."
        )


def _validate_error_model_coefficients(
    error_model: str, ar_coefficients: Any
) -> None:
    _validate_error_model(error_model)
    _validate_ar_coefficients(ar_coefficients)
    n_coefficients = len(np.asarray(ar_coefficients))
    expected = _ERROR_MODEL_ORDERS[error_model]
    if n_coefficients != expected:
        raise InvalidConfigurationError(
            f"error_model={error_model!r} requires {expected} AR coefficients, "
            f"got {n_coefficients}."
        )


def _validate_error_model(error_model: str) -> None:
    if error_model not in _ERROR_MODEL_ORDERS:
        raise InvalidConfigurationError(
            f"Unknown error_model: {error_model!r}. "
            f"Expected one of {sorted(_ERROR_MODEL_ORDERS)}."
        )


def _validate_stochastic_cycle_mode(stochastic_cycle_mode: str) -> None:
    if stochastic_cycle_mode not in _STOCHASTIC_CYCLE_MODES:
        raise InvalidConfigurationError(
            f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}. "
            f"Expected one of {sorted(_STOCHASTIC_CYCLE_MODES)}."
        )


def _validate_ar_spectral_adjustment_values(
    ar_spectral_adjustment: Any, expected_shape: tuple
) -> None:
    _validate_numeric_1d_array(
        ar_spectral_adjustment, "ar_spectral_adjustment"
    )
    adjustment = np.asarray(ar_spectral_adjustment, dtype=float)
    if adjustment.shape != expected_shape:
        raise InvalidConfigurationError(
            f"ar_spectral_adjustment must have shape {expected_shape}, "
            f"got {adjustment.shape}."
        )
    if np.any(adjustment <= 0.0):
        raise InvalidConfigurationError(
            "ar_spectral_adjustment values must all be positive."
        )
