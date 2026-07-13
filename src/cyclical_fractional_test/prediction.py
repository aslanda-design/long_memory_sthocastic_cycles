from __future__ import annotations

from typing import Sequence

import numpy as np

from .exceptions import InvalidConfigurationError
from .filters import compute_fractional_coefficients_from_mu, compute_mu

# ---------------------------------------------------------------------------
# Generative model used for prediction
#
#   Y_t              = X_t·β + S_t                     (deterministic + stochastic)
#   filter_{R,D}(S)_t = ε_t                            (cyclic long memory)
#   ε_t              = Σ_i φ_i·ε_{t-i} + e_t           (AR(p) error, p∈{0,1,2})
#
# where ε̂ are exactly the filtered-regression residuals, β the OLS coefficients,
# and φ the AR nuisance coefficients estimated from those residuals. The one-step
# conditional mean gives the in-sample reconstruction; the same recursion, fed
# with AR-forecast innovations, produces the out-of-sample forecast.
# ---------------------------------------------------------------------------

_SINGLE_MODES = {"single", "multi_peak_single_cycle"}


def reconstruct_in_sample(
    y: np.ndarray,
    X: np.ndarray,
    cycles: Sequence,
    betas: np.ndarray,
    residuals: np.ndarray,
    ar_coefficients: np.ndarray,
    mode: str = "single",
) -> np.ndarray:
    """Return the in-sample one-step-ahead reconstruction ŷ_t for t = 1, ..., T.

    ŷ_t = X_t·β̂  −  Σ_{j=1}^{t-1} C_j·S_{t-j}  +  Σ_{i=1}^{p} φ_i·ε̂_{t-i}

    with S_t = Y_t − X_t·β̂. By construction Y_t − ŷ_t equals the AR innovation e_t.
    """
    _validate_reconstruct_in_sample(y, X, cycles, betas, residuals, ar_coefficients, mode)

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    betas_arr = np.asarray(betas, dtype=float)
    eps = np.asarray(residuals, dtype=float)
    phi = np.asarray(ar_coefficients, dtype=float)
    p = len(phi)
    T = len(y_arr)

    deterministic = X_arr.dot(betas_arr)
    S = y_arr - deterministic
    C = _combined_coefficients(cycles, T, T, mode)

    yhat = np.empty(T, dtype=float)
    for a in range(T):
        # − Σ_{j=1}^{a} C_j S_{a-j}: stochastic memory carried from the past.
        stochastic = -np.dot(C[1 : a + 1], S[a - 1 :: -1]) if a > 0 else 0.0
        ar = sum(phi[i - 1] * eps[a - i] for i in range(1, p + 1) if a - i >= 0)
        yhat[a] = deterministic[a] + stochastic + ar
    return yhat


def forecast_ar(
    residuals: np.ndarray,
    ar_coefficients: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Forecast the AR(p) error process `horizon` steps beyond the sample.

    ε̂_{T+k} = Σ_i φ_i·ε̂_{T+k-i}, taking the expected innovation e to be zero.
    For white noise (no coefficients) the forecast is zero at every horizon.
    """
    _validate_forecast_ar(residuals, ar_coefficients, horizon)

    eps = np.asarray(residuals, dtype=float)
    phi = np.asarray(ar_coefficients, dtype=float)
    p = len(phi)
    T = len(eps)

    full = np.concatenate([eps, np.zeros(horizon, dtype=float)])
    for k in range(horizon):
        a = T + k
        full[a] = sum(phi[i - 1] * full[a - i] for i in range(1, p + 1))
    return full[T:]


def forecast_out_of_sample(
    y: np.ndarray,
    X: np.ndarray,
    X_future: np.ndarray,
    cycles: Sequence,
    betas: np.ndarray,
    residuals: np.ndarray,
    ar_coefficients: np.ndarray,
    mode: str = "single",
    horizon: int | None = None,
    coefficient_t_ref: int | None = None,
) -> np.ndarray:
    """Forecast ŷ_t for t = T+1, ..., T+horizon.

    The AR error is forecast first; the stochastic component is then propagated by
    inverting the cyclic filter, S_a = ε̂_a − Σ_{j=1}^{a} C_j·S_{a-j}, reusing the
    in-sample S values for the known indices. The deterministic part X_future·β̂
    extrapolates the Chebyshev basis with the training length held fixed.

    `coefficient_t_ref` optionally fixes the cyclic-filter reference length used
    to compute mu. The default is len(y), which is the usual out-of-sample case.
    Use a larger value when forecasting recursively from a cutoff inside a model
    that was fitted on a longer sample.
    """
    h = X_future.shape[0] if horizon is None else horizon
    _validate_forecast_out_of_sample(
        y,
        X,
        X_future,
        cycles,
        betas,
        residuals,
        ar_coefficients,
        mode,
        h,
        coefficient_t_ref,
    )

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    X_future_arr = np.asarray(X_future, dtype=float)
    betas_arr = np.asarray(betas, dtype=float)
    eps = np.asarray(residuals, dtype=float)
    phi = np.asarray(ar_coefficients, dtype=float)
    p = len(phi)
    T = len(y_arr)
    t_ref = T if coefficient_t_ref is None else int(coefficient_t_ref)

    S_known = y_arr - X_arr.dot(betas_arr)
    deterministic_future = X_future_arr.dot(betas_arr)
    C = _combined_coefficients(cycles, t_ref, T + h, mode)

    eps_full = np.concatenate([eps, np.zeros(h, dtype=float)])
    S_full = np.concatenate([S_known, np.zeros(h, dtype=float)])
    for k in range(h):
        a = T + k
        eps_full[a] = sum(phi[i - 1] * eps_full[a - i] for i in range(1, p + 1))
        # S_a = ε̂_a − Σ_{j=1}^{a} C_j S_{a-j}: invert the cyclic filter forward.
        carried = np.dot(C[1 : a + 1], S_full[a - 1 :: -1])
        S_full[a] = eps_full[a] - carried

    return deterministic_future + S_full[T:]


def compute_ma_weights(
    cycles: Sequence,
    ar_coefficients: np.ndarray,
    mode: str,
    T_ref: int,
    length: int,
) -> np.ndarray:
    """Return the MA(∞) weights ψ_l mapping innovations e_t to the stochastic part S_t.

    S = filter_{R,D}^{-1}(ε) and ε = AR(p)^{-1}(e), so ψ is the convolution of the
    inverse cyclic filter coefficients (params −D) with the AR(p) inverse MA weights.
    The horizon-k forecast-error variance is σ² Σ_{l=0}^{k-1} ψ_l².
    """
    _validate_compute_ma_weights(cycles, ar_coefficients, mode, T_ref, length)

    cycles_t = tuple(cycles)
    inverse_filter = np.zeros(length, dtype=float)
    inverse_filter[0] = 1.0
    relevant = cycles_t[:1] if mode in _SINGLE_MODES else cycles_t
    for cycle in relevant:
        mu = compute_mu(T_ref, cycle.R)
        coeffs = compute_fractional_coefficients_from_mu(mu, -cycle.D, length)
        inverse_filter = np.convolve(inverse_filter, coeffs, mode="full")[:length]

    phi = np.asarray(ar_coefficients, dtype=float)
    p = len(phi)
    psi_ar = np.zeros(length, dtype=float)
    psi_ar[0] = 1.0
    for l in range(1, length):
        psi_ar[l] = sum(phi[i - 1] * psi_ar[l - i] for i in range(1, p + 1) if l - i >= 0)

    return np.convolve(inverse_filter, psi_ar, mode="full")[:length]


def _combined_coefficients(
    cycles: Sequence,
    T_ref: int,
    length: int,
    mode: str,
) -> np.ndarray:
    """Coefficients of the (possibly multi-cycle) filter, with mu fixed at T_ref.

    Single-cycle modes use the one cycle directly; multi_cycle convolves the
    per-cycle coefficient arrays (truncated to `length`).
    """
    cycles_t = tuple(cycles)
    if mode in _SINGLE_MODES:
        cycle = cycles_t[0]
        mu = compute_mu(T_ref, cycle.R)
        return compute_fractional_coefficients_from_mu(mu, cycle.D, length)

    combined = np.zeros(length, dtype=float)
    combined[0] = 1.0
    for cycle in cycles_t:
        mu = compute_mu(T_ref, cycle.R)
        cycle_coeffs = compute_fractional_coefficients_from_mu(mu, cycle.D, length)
        combined = np.convolve(combined, cycle_coeffs, mode="full")[:length]
    return combined


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know the exact
# reason why the process failed.
# ---------------------------------------------------------------------------

_VALID_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}


def _validate_cycles_and_mode(cycles: object, mode: object) -> None:
    if mode not in _VALID_MODES:
        raise InvalidConfigurationError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(_VALID_MODES)}."
        )
    try:
        cycles_t = tuple(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycles_t) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")
    if mode in _SINGLE_MODES and len(cycles_t) != 1:
        raise InvalidConfigurationError(
            f"mode={mode!r} requires exactly 1 cycle, got {len(cycles_t)}."
        )


def _validate_design_and_betas(
    y_arr: np.ndarray, X: object, betas: object, residuals: object
) -> None:
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
    try:
        betas_arr = np.asarray(betas, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"betas must be numeric: {exc}") from exc
    if betas_arr.ndim != 1 or len(betas_arr) != X_arr.shape[1]:
        raise InvalidConfigurationError(
            f"betas must be 1-D of length {X_arr.shape[1]}, got shape {betas_arr.shape}."
        )
    try:
        eps = np.asarray(residuals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"residuals must be numeric: {exc}") from exc
    if eps.ndim != 1 or len(eps) != len(y_arr):
        raise InvalidConfigurationError(
            f"residuals must be 1-D of length {len(y_arr)}, got shape {eps.shape}."
        )


def _validate_y(y: object) -> np.ndarray:
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"y must be numeric: {exc}") from exc
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise InvalidConfigurationError("y must be a non-empty 1-D array.")
    return y_arr


def _validate_ar_coefficients(ar_coefficients: object) -> None:
    try:
        coeffs = np.asarray(ar_coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"ar_coefficients must be numeric: {exc}"
        ) from exc
    if coeffs.ndim != 1:
        raise InvalidConfigurationError("ar_coefficients must be a 1-D array.")
    if not np.all(np.isfinite(coeffs)):
        raise InvalidConfigurationError("ar_coefficients contains non-finite values.")


def _validate_horizon(horizon: object) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise InvalidConfigurationError(
            f"horizon must be an int, got {type(horizon).__name__}."
        )
    if horizon < 1:
        raise InvalidConfigurationError(f"horizon must be >= 1, got {horizon}.")


def _validate_compute_ma_weights(
    cycles, ar_coefficients, mode, T_ref, length
) -> None:
    _validate_cycles_and_mode(cycles, mode)
    _validate_ar_coefficients(ar_coefficients)
    if isinstance(T_ref, bool) or not isinstance(T_ref, int) or T_ref < 2:
        raise InvalidConfigurationError(f"T_ref must be an int >= 2, got {T_ref!r}.")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise InvalidConfigurationError(f"length must be an int >= 1, got {length!r}.")


def _validate_reconstruct_in_sample(
    y, X, cycles, betas, residuals, ar_coefficients, mode
) -> None:
    y_arr = _validate_y(y)
    _validate_cycles_and_mode(cycles, mode)
    _validate_design_and_betas(y_arr, X, betas, residuals)
    _validate_ar_coefficients(ar_coefficients)


def _validate_forecast_ar(residuals, ar_coefficients, horizon) -> None:
    try:
        eps = np.asarray(residuals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"residuals must be numeric: {exc}") from exc
    if eps.ndim != 1 or eps.size == 0:
        raise InvalidConfigurationError("residuals must be a non-empty 1-D array.")
    _validate_ar_coefficients(ar_coefficients)
    if len(np.asarray(ar_coefficients)) >= len(eps):
        raise InvalidConfigurationError(
            "residuals must be longer than the AR order."
        )
    _validate_horizon(horizon)


def _validate_forecast_out_of_sample(
    y,
    X,
    X_future,
    cycles,
    betas,
    residuals,
    ar_coefficients,
    mode,
    horizon,
    coefficient_t_ref=None,
) -> None:
    y_arr = _validate_y(y)
    _validate_cycles_and_mode(cycles, mode)
    _validate_design_and_betas(y_arr, X, betas, residuals)
    _validate_ar_coefficients(ar_coefficients)
    _validate_horizon(horizon)
    if coefficient_t_ref is not None:
        if (
            isinstance(coefficient_t_ref, bool)
            or not isinstance(coefficient_t_ref, (int, np.integer))
            or int(coefficient_t_ref) < len(y_arr)
        ):
            raise InvalidConfigurationError(
                "coefficient_t_ref must be an int greater than or equal to len(y), "
                f"got {coefficient_t_ref!r} for len(y)={len(y_arr)}."
            )
    try:
        X_future_arr = np.asarray(X_future, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"X_future must be numeric: {exc}") from exc
    if X_future_arr.ndim != 2:
        raise InvalidConfigurationError(
            f"X_future must be 2-D, got shape {X_future_arr.shape}."
        )
    if X_future_arr.shape[0] != horizon:
        raise InvalidConfigurationError(
            f"X_future must have {horizon} rows, got {X_future_arr.shape[0]}."
        )
    if X_future_arr.shape[1] != np.asarray(X, dtype=float).shape[1]:
        raise InvalidConfigurationError(
            "X_future must have the same number of columns as X."
        )
