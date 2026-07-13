from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .exceptions import InvalidConfigurationError, InvalidCycleError


def compute_mu(T: int, R: int) -> float:
    """Compute mu = cos(2πR/T), the frequency cosine."""
    _validate_compute_mu(T, R)
    return float(np.cos(2.0 * np.pi * R / T))


def compute_fractional_coefficients_from_mu(
    mu: float,
    D: float,
    length: int,
) -> np.ndarray:
    """Compute the first `length` coefficients C_{j,D}(mu) of (1 - 2 mu L + L^2)^D.

    C_0 = 1,  C_1 = -2 D mu,
    C_j = [2 mu (j - 1 - D) C_{j-1} + (2D - j + 2) C_{j-2}] / j  for j >= 2.

    The recursion depends on the frequency only through mu, so `length` can exceed
    the training length — used to extend the filter for out-of-sample prediction.
    """
    _validate_compute_fractional_coefficients_from_mu(mu, D, length)
    coeffs = np.zeros(length, dtype=float)
    coeffs[0] = 1.0
    if length == 1:
        return coeffs
    coeffs[1] = -2.0 * D * mu
    for j in range(2, length):
        coeffs[j] = (
            2.0 * mu * (j - 1 - D) * coeffs[j - 1]
            + (2.0 * D - j + 2) * coeffs[j - 2]
        ) / j
    return coeffs


def compute_fractional_coefficients_single_cycle(
    T: int,
    R: int,
    D: float,
) -> np.ndarray:
    """Compute coefficients C_{j,D}(mu) of (1 - 2 mu L + L^2)^D for j = 0, ..., T-1."""
    _validate_compute_fractional_coefficients_single_cycle(T, R, D)
    mu = compute_mu(T, R)
    return compute_fractional_coefficients_from_mu(mu, D, T)


def compute_fractional_coefficients_multi_cycle(
    T: int,
    cycles: object,
) -> np.ndarray:
    """Compute coefficients of Π_q (1 - 2 mu_q L + L²)^Dq.

    The combined coefficients are the truncated convolution of the single-cycle
    coefficient arrays, keeping C_0, ..., C_{T-1}.
    """
    try:
        cycles_t = tuple(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    _validate_compute_fractional_coefficients_multi_cycle(T, cycles_t)
    combined = np.zeros(T, dtype=float)
    combined[0] = 1.0
    for cycle in cycles_t:
        cycle_coeffs = compute_fractional_coefficients_single_cycle(
            T, cycle.R, cycle.D
        )
        combined = np.convolve(combined, cycle_coeffs, mode="full")[:T]
    return combined


def compute_fractional_coefficients_dynamic(
    T: int,
    cycles: Sequence,
    mode: str = "single",
) -> np.ndarray:
    """Dispatch to the single- or multi-cycle coefficient computation based on mode.

    "multi_peak_single_cycle" uses the single-cycle path; peak selection already happened upstream.
    """

    _validate_compute_fractional_coefficients_dynamic(cycles, mode)

    if mode in ("single", "multi_peak_single_cycle"):
        cycle = cycles[0]
        return compute_fractional_coefficients_single_cycle(T, cycle.R, cycle.D)
    return compute_fractional_coefficients_multi_cycle(T, cycles)


def apply_fractional_filter_single_series(
    x: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply the causal convolution x_filtered[t] = sum_{j=0}^{t} C_j x[t-j]."""

    _validate_apply_fractional_filter_single_series(x, coefficients)
    
    x_arr = np.asarray(x, dtype=float)
    c_arr = np.asarray(coefficients, dtype=float)
    return np.convolve(x_arr, c_arr, mode="full")[: len(x_arr)]


def apply_single_cycle_filter(
    x: np.ndarray,
    cycle: object,
    T: int,
) -> np.ndarray:
    """Filter x with (1 - 2 mu L + L^2)^D using the coefficients from Wave 6."""
    _validate_apply_single_cycle_filter(x, T)
    coefficients = compute_fractional_coefficients_single_cycle(T, cycle.R, cycle.D)
    return apply_fractional_filter_single_series(x, coefficients)


def apply_multi_cycle_filter(
    x: np.ndarray,
    cycles: object,
    T: int,
) -> np.ndarray:
    """Apply Π_q (1 - 2 mu_q L + L^2)^Dq by chaining single-cycle filters sequentially."""
    _validate_apply_multi_cycle_filter(x, cycles, T)
    result = np.asarray(x, dtype=float).copy()
    for cycle in cycles:
        result = apply_single_cycle_filter(result, cycle, T)
    return result


def apply_filter_dynamic(
    x: np.ndarray,
    cycles: Sequence,
    T: int,
    mode: str = "single",
) -> np.ndarray:
    """Dispatch to the single- or multi-cycle filter based on mode.

    "multi_peak_single_cycle" uses the single-cycle path; peak selection already happened upstream.
    """
    _validate_apply_filter_dynamic(cycles, mode)
    if mode in ("single", "multi_peak_single_cycle"):
        return apply_single_cycle_filter(x, cycles[0], T)
    return apply_multi_cycle_filter(x, cycles, T)


def filter_response_and_design(
    y: np.ndarray,
    X: np.ndarray,
    cycles: Sequence,
    mode: str = "single",
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter y and each column of X with the fractional cyclic filter.

    Returns (y_filtered, X_filtered) preserving the shapes (T,) and (T, p).
    """
    _validate_filter_response_and_design(y, X)
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    T = len(y_arr)
    y_filtered = apply_filter_dynamic(y_arr, cycles, T, mode)
    if X_arr.shape[1] == 0:
        X_filtered = np.empty((T, 0), dtype=float)
    else:
        X_filtered = np.column_stack([
            apply_filter_dynamic(X_arr[:, k], cycles, T, mode)
            for k in range(X_arr.shape[1])
        ])
    return y_filtered, X_filtered


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_compute_mu(T: int, R: int) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    if isinstance(R, bool) or not isinstance(R, int):
        raise InvalidConfigurationError(f"R must be an int, got {type(R).__name__}.")
    if R < 0 or R >= T:
        raise InvalidConfigurationError(
            f"R must satisfy 0 <= R < T={T}, got R={R}."
        )


def _validate_compute_fractional_coefficients_from_mu(
    mu: float, D: float, length: int
) -> None:
    if isinstance(mu, bool):
        raise InvalidConfigurationError("mu must not be a bool.")
    try:
        mu_val = float(mu)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"mu must be numeric: {exc}") from exc
    if not np.isfinite(mu_val):
        raise InvalidConfigurationError(f"mu must be finite, got {mu}.")
    if isinstance(D, bool):
        raise InvalidConfigurationError("D must not be a bool.")
    try:
        D_val = float(D)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"D must be a numeric value: {exc}") from exc
    if not np.isfinite(D_val):
        raise InvalidConfigurationError(f"D must be finite, got {D}.")
    if isinstance(length, bool) or not isinstance(length, int):
        raise InvalidConfigurationError(
            f"length must be an int, got {type(length).__name__}."
        )
    if length < 1:
        raise InvalidConfigurationError(f"length must be >= 1, got {length}.")


def _validate_compute_fractional_coefficients_single_cycle(
    T: int, R: int, D: float
) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    if isinstance(R, bool) or not isinstance(R, int):
        raise InvalidConfigurationError(f"R must be an int, got {type(R).__name__}.")
    if R < 0 or R >= T:
        raise InvalidConfigurationError(
            f"R must satisfy 0 <= R < T={T}, got R={R}."
        )
    if isinstance(D, bool):
        raise InvalidConfigurationError("D must not be a bool.")
    try:
        D_val = float(D)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"D must be a numeric value: {exc}") from exc
    if not np.isfinite(D_val):
        raise InvalidConfigurationError(f"D must be finite, got {D}.")


def _validate_compute_fractional_coefficients_multi_cycle(
    T: int, cycles: object
) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")
    for cycle in cycle_list:
        _validate_compute_fractional_coefficients_single_cycle(
            T, cycle.R, cycle.D
        )


def _validate_compute_fractional_coefficients_dynamic(
    cycles: Sequence, mode: str
) -> None:
    _VALID_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}
    if mode not in _VALID_MODES:
        raise InvalidConfigurationError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(_VALID_MODES)}."
        )
    if mode in ("single", "multi_peak_single_cycle") and len(cycles) != 1:
        raise InvalidCycleError(
            f"mode={mode!r} requires exactly 1 cycle, got {len(cycles)}."
        )


def _validate_apply_fractional_filter_single_series(
    x: object, coefficients: object
) -> None:
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"x must be numeric: {exc}") from exc
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise InvalidConfigurationError("x must be a non-empty 1-D array.")
    try:
        c_arr = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"coefficients must be numeric: {exc}") from exc
    if c_arr.ndim != 1 or c_arr.size == 0:
        raise InvalidConfigurationError("coefficients must be a non-empty 1-D array.")
    if len(c_arr) < len(x_arr):
        raise InvalidConfigurationError(
            f"len(coefficients)={len(c_arr)} must be >= len(x)={len(x_arr)}."
        )


def _validate_apply_single_cycle_filter(x: object, T: int) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"x must be numeric: {exc}") from exc
    if x_arr.ndim != 1:
        raise InvalidConfigurationError("x must be a 1-D array.")
    if len(x_arr) != T:
        raise InvalidConfigurationError(f"len(x)={len(x_arr)} must equal T={T}.")


def _validate_apply_multi_cycle_filter(x: object, cycles: object, T: int) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"x must be numeric: {exc}") from exc
    if x_arr.ndim != 1 or len(x_arr) != T:
        raise InvalidConfigurationError("x must be a 1-D array of length T.")
    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")


def _validate_apply_filter_dynamic(cycles: Sequence, mode: str) -> None:
    _VALID_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}
    if mode not in _VALID_MODES:
        raise InvalidConfigurationError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(_VALID_MODES)}."
        )
    if mode in ("single", "multi_peak_single_cycle") and len(cycles) != 1:
        raise InvalidCycleError(
            f"mode={mode!r} requires exactly 1 cycle, got {len(cycles)}."
        )


def _validate_filter_response_and_design(y: object, X: object) -> None:
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"y must be numeric: {exc}") from exc
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise InvalidConfigurationError("y must be a non-empty 1-D array.")
    try:
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"X must be numeric: {exc}") from exc
    if X_arr.ndim != 2:
        raise InvalidConfigurationError(f"X must be 2-D, got shape {X_arr.shape}.")
    if X_arr.shape[0] != len(y_arr):
        raise InvalidConfigurationError(
            f"X.shape[0]={X_arr.shape[0]} must equal len(y)={len(y_arr)}."
        )
