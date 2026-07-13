from __future__ import annotations

from typing import Sequence

import numpy as np

from .exceptions import InvalidConfigurationError, InvalidCycleError
from .filters import filter_response_and_design
from .grid import build_d_fine_grid, build_d_grid_for_strategy
from .regression import compute_time_variance, estimate_ar_ols, fit_filtered_regression
from .results import AdaptiveDSearchResult, GridCandidateResult, StochasticCycle
from .scoring import compute_test_star_statistic, compute_test_statistic, score_candidate
from .spectral import (
    compute_frequency_variance_dynamic,
    compute_ar_spectral_adjustment,
    compute_psi_dynamic,
    compute_residual_periodogram,
    compute_xa_error_model,
    compute_xaa_error_model,
)
from .validation import validate_config

_ERROR_MODEL_ORDERS = {"white_noise": 0, "ar1": 1, "ar2": 2}
_FREQUENCY_STATISTIC_MODES = {"frequency", "test_star"}


def evaluate_candidate(
    y: np.ndarray,
    X: np.ndarray,
    cycles: Sequence[StochasticCycle],
    config: object,
) -> GridCandidateResult:
    """Evaluate one grid candidate and return its full set of statistics.

    Orchestrates Waves 5–11 and 16 for a single (cycles, config) combination.
    Does not build grids, select R*, perform scoring, or rank results.
    """
    _validate_evaluate_candidate(y, X, cycles, config)

    mode = config.stochastic_cycle_mode
    cycles_t = tuple(cycles)
    T = len(y)

    psi = compute_psi_dynamic(T, cycles_t, mode, config.drop_singular_frequency)

    y_f, X_f = filter_response_and_design(y, X, cycles_t, mode=mode)
    reg = fit_filtered_regression(y_f, X_f)

    lambdas_resid, I_resid = compute_residual_periodogram(reg.residuals)
    variance_time = compute_time_variance(reg.residuals)
    ar_coefficients = estimate_ar_ols(
        reg.residuals, _ERROR_MODEL_ORDERS[config.error_model]
    )
    ar_spectral_adjustment = compute_ar_spectral_adjustment(
        lambdas_resid, ar_coefficients
    )
    xaa = compute_xaa_error_model(
        psi, lambdas_resid, config.error_model, ar_coefficients, mode
    )
    xa = compute_xa_error_model(
        psi, I_resid, config.error_model, ar_spectral_adjustment, mode
    )
    variance_frequency = compute_frequency_variance_dynamic(
        I_resid, cycles_t, mode=mode, drop_frequency=config.drop_singular_frequency
    )

    test_value = compute_test_statistic(T, xa, xaa, variance_time)
    test_star_value = _compute_test_star_for_candidate(
        T,
        xa,
        xaa,
        variance_frequency,
        config.statistic_mode,
    )

    return GridCandidateResult(
        cycles=cycles_t,
        error_model=config.error_model,
        ar_coefficients=tuple(float(value) for value in ar_coefficients),
        test_value=test_value,
        test_star_value=test_star_value,
        abs_test_value=abs(test_value),
        abs_test_star_value=abs(test_star_value),
        xa=xa,
        xaa=xaa,
        variance_time=variance_time,
        variance_frequency=variance_frequency,
        betas=reg.betas,
        beta_standard_errors=reg.beta_standard_errors,
        residuals=reg.residuals,
        residual_sum_squares=reg.residual_sum_squares,
    )


def _compute_test_star_for_candidate(
    T: int,
    xa: float,
    xaa: float,
    variance_frequency: float,
    statistic_mode: str,
) -> float:
    """Return TEST* when VAR* is usable; otherwise leave it undefined for TEST ranking."""
    if variance_frequency > np.finfo(float).eps:
        return compute_test_star_statistic(T, xa, xaa, variance_frequency)
    if statistic_mode in _FREQUENCY_STATISTIC_MODES:
        raise InvalidConfigurationError(
            "TEST* is undefined because variance_frequency is numerically zero; "
            "use statistic_mode='test' for this degenerate candidate."
        )
    return float("nan")


def evaluate_r_with_adaptive_d(
    y: np.ndarray,
    X: np.ndarray,
    R: int,
    config: object,
) -> AdaptiveDSearchResult:
    """Run the coarse-to-fine D search for one frequency index R.

    Evaluates the coarse grid, refines locally around the best coarse D, and
    returns the best candidate found. The fine grid reuses the coarse result
    for any D already evaluated, so each (R,D) pair is computed at most once.
    """
    _validate_evaluate_r_with_adaptive_d(y, X, R, config)

    mode = config.statistic_mode
    coarse_grid = build_d_grid_for_strategy(config)  # adaptive → coarse grid

    # Evaluate the coarse stage; keep one result per distinct D value.
    results_by_d: dict[float, GridCandidateResult] = {}
    for d in coarse_grid:
        key = round(float(d), 12)
        if key in results_by_d:
            continue
        results_by_d[key] = evaluate_candidate(
            y, X, (StochasticCycle(R=int(R), D=key),), config
        )
    n_coarse = len(results_by_d)

    best_coarse_d = min(results_by_d, key=lambda k: score_candidate(results_by_d[k], mode))
    best_coarse_result = results_by_d[best_coarse_d]

    # Refine locally; only evaluate D values not already seen in the coarse stage.
    fine_grid = build_d_fine_grid(best_coarse_d, config.d_fine_radius, config.d_fine_step)
    n_fine = 0
    for d in fine_grid:
        key = round(float(d), 12)
        if key in results_by_d:
            continue
        results_by_d[key] = evaluate_candidate(
            y, X, (StochasticCycle(R=int(R), D=key),), config
        )
        n_fine += 1

    best_d = min(results_by_d, key=lambda k: score_candidate(results_by_d[k], mode))
    best_result = results_by_d[best_d]

    return AdaptiveDSearchResult(
        R=int(R),
        best_result=best_result,
        best_coarse_result=best_coarse_result,
        best_coarse_d=float(best_coarse_d),
        best_d=float(best_d),
        n_coarse_evaluated=n_coarse,
        n_fine_evaluated=n_fine,
        n_candidates_evaluated=len(results_by_d),
        all_results=list(results_by_d.values()),
    )


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know the exact
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_evaluate_candidate(
    y: object, X: object, cycles: object, config: object
) -> None:
    validate_config(config)
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
        raise InvalidConfigurationError(
            f"X must be 2-D, got shape {X_arr.shape}."
        )
    if X_arr.shape[0] != len(y_arr):
        raise InvalidConfigurationError(
            f"X.shape[0]={X_arr.shape[0]} must equal len(y)={len(y_arr)}."
        )

    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")

    mode = getattr(config, "stochastic_cycle_mode", None)
    if mode == "single" and len(cycle_list) != 1:
        raise InvalidCycleError(
            f"stochastic_cycle_mode='single' requires exactly 1 cycle, "
            f"got {len(cycle_list)}."
        )


def _validate_evaluate_r_with_adaptive_d(
    y: object, X: object, R: object, config: object
) -> None:
    validate_config(config)
    if isinstance(R, bool) or not isinstance(R, (int, np.integer)):
        raise InvalidCycleError(f"R must be an int, got {type(R).__name__}.")
    if int(R) < 0:
        raise InvalidCycleError(f"R must be >= 0, got {R}.")
