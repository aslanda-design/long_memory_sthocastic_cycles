from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from .chebyshev import build_chebyshev_design
from .config import CyclicalTestConfig
from .diagnostics import build_test_diagnostics
from .evaluation import evaluate_candidate, evaluate_r_with_adaptive_d
from .exceptions import InvalidConfigurationError
from .grid import (
    build_d_fine_grid,
    build_d_grid_for_strategy,
    build_multi_cycle_candidate_grid,
    build_multi_cycle_candidate_grid_from_d_grids,
    build_r_grid_around_peak,
    build_single_cycle_candidate_grid,
)
from .results import CyclicalFractionalTestResult, GridCandidateResult, StochasticCycle
from .scoring import TopKSelector, score_candidate
from .spectral import (
    compute_autocorrelogram as compute_series_autocorrelogram,
    compute_document_periodogram,
)
from .validation import validate_config, validate_series


def compute_periodogram(y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return the periodogram of a time series.

    Returns (lambdas, I_y) where lambdas are the Fourier frequencies and
    I_y are the corresponding periodogram values, normalised as I(λ_j) = |FFT|²/(2πT).
    """
    arr = validate_series(y)
    return compute_document_periodogram(arr)


def compute_autocorrelogram(
    y: Any,
    max_lag: int | None = None,
    adjusted: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the autocorrelogram of a time series.

    Returns (lags, autocorrelations), where lags are 0, ..., max_lag and
    autocorrelations are sample autocorrelations of the demeaned series.
    """
    return compute_series_autocorrelogram(y, max_lag=max_lag, adjusted=adjusted)


def run_cyclical_fractional_test(
    y: Any,
    config: Optional[CyclicalTestConfig] = None,
    threshold: Optional[float] = None,
    **kwargs: Any,
) -> CyclicalFractionalTestResult:
    """Run the fractional cyclic long-memory test.

    If config is None, CyclicalTestConfig defaults are used.
    kwargs override individual config fields (e.g., top_k=3 or error_model="ar1").
    stochastic_cycle_mode='single' searches one cycle. stochastic_cycle_mode='multi_cycle'
    selects the top n_stochastic_cycles periodogram peaks and evaluates joint
    fixed-grid or adaptive Cartesian products of D values for those simultaneous cycles.
    config.ignored_stochastic_rs excludes known frequency indices from stochastic
    cycle selection. config.chebyshev_orders can replace the contiguous
    deterministic Chebyshev basis with explicit positive polynomial orders.
    The result includes a TestDiagnostics object and per-candidate AR nuisance estimates.

    If threshold is given (a positive float), the result's under_threshold_results
    collects every evaluated candidate whose statistic score (|TEST| or |TEST*|,
    per config.statistic_mode) falls below it. Results are grouped by the full
    frequency tuple: (R,) for single-cycle candidates and (R1, R2, ...) for
    multi-cycle candidates. When threshold is None this object stays None.
    """
    if config is None:
        config = CyclicalTestConfig()
    if kwargs:
        config = dataclasses.replace(config, **kwargs)

    arr = validate_series(y)
    validate_config(config)

    threshold_value = _validate_threshold(threshold) if threshold is not None else None

    if config.stochastic_cycle_mode == "multi_peak_single_cycle":
        raise NotImplementedError(
            f"stochastic_cycle_mode={config.stochastic_cycle_mode!r} is not yet "
            "supported in run_cyclical_fractional_test. "
            "Use stochastic_cycle_mode='single' or 'multi_cycle'."
        )

    T = len(arr)
    X = build_chebyshev_design(
        T,
        config.n_deterministic_cycles,
        config.include_intercept,
        config.chebyshev_orders,
    )

    lambdas_y, I_y = compute_document_periodogram(arr)
    periodogram = I_y[:len(I_y) // 2]
    ignored_rs = _build_ignored_r_set(config, T)

    # The reported grid is the full fixed grid, or the coarse seed for adaptive search.
    d_grid = build_d_grid_for_strategy(config)

    if config.stochastic_cycle_mode == "multi_cycle":
        return _run_multi_cycle_fractional_test(
            arr=arr,
            X=X,
            lambdas_y=lambdas_y,
            I_y=I_y,
            periodogram=periodogram,
            d_grid=d_grid,
            config=config,
            threshold_value=threshold_value,
            ignored_rs=ignored_rs,
        )

    r_peak = _find_periodogram_peak_excluding(periodogram, ignored_rs)

    r_candidates = build_r_grid_around_peak(
        r_peak,
        config.r_window,
        T,
        include_zero=not config.exclude_zero_frequency,
    )
    r_candidates = _filter_ignored_r_candidates(r_candidates, ignored_rs)

    selector = TopKSelector(k=config.top_k, statistic_mode=config.statistic_mode)

    # When a threshold is requested, collect every evaluated candidate that scores
    # below it, grouped by full frequency tuple; otherwise leave it as None.
    under_threshold_results: Optional[Dict[Tuple[int, ...], List[GridCandidateResult]]] = (
        {} if threshold_value is not None else None
    )

    n_evaluated = 0
    warnings: List[str] = []
    adaptive_info: Optional[dict] = None

    if config.d_search_strategy == "fixed_grid":
        for cycles in build_single_cycle_candidate_grid(r_candidates, d_grid):
            candidate_result = evaluate_candidate(arr, X, cycles, config)
            selector.consider(candidate_result)
            if under_threshold_results is not None:
                _record_if_under_threshold(
                    under_threshold_results,
                    candidate_result,
                    threshold_value,
                    config.statistic_mode,
                    config.return_residuals_for_threshold,
                )
            n_evaluated += 1
            logger.info(
                "candidate R=%d D=%.2f  XA=%.6f",
                cycles[0].R,
                cycles[0].D,
                candidate_result.xa,
            )
    else:
        # Adaptive coarse-to-fine search: one best candidate per frequency R.
        best_coarse_d: List[float] = []
        final_d: List[float] = []
        n_coarse = 0
        n_fine = 0
        for R in r_candidates:
            search = evaluate_r_with_adaptive_d(arr, X, int(R), config)
            selector.consider(search.best_result)
            if under_threshold_results is not None:
                for candidate in search.all_results:
                    _record_if_under_threshold(
                        under_threshold_results,
                        candidate,
                        threshold_value,
                        config.statistic_mode,
                        config.return_residuals_for_threshold,
                    )
            n_evaluated += search.n_candidates_evaluated
            n_coarse += search.n_coarse_evaluated
            n_fine += search.n_fine_evaluated
            best_coarse_d.append(search.best_coarse_d)
            final_d.append(search.best_d)
            logger.info(
                "R=%d adaptive best D=%.2f (coarse D=%.2f, %d candidates)",
                int(R),
                search.best_d,
                search.best_coarse_d,
                search.n_candidates_evaluated,
            )
        adaptive_info = {
            "d_coarse_grid": d_grid,
            "d_fine_step": config.d_fine_step,
            "d_fine_radius": config.d_fine_radius,
            "best_coarse_d_per_r": best_coarse_d,
            "final_d_per_r": final_d,
            "n_coarse_evaluations": n_coarse,
            "n_fine_evaluations": n_fine,
        }

    n_valid = n_evaluated
    top_k_results = selector.get_top_k()
    best_result = selector.get_best()

    if under_threshold_results is not None:
        # Order R tuples ascending and list passing candidates best-first.
        under_threshold_results = {
            R: sorted(candidates, key=lambda c: score_candidate(c, config.statistic_mode))
            for R, candidates in sorted(under_threshold_results.items())
        }

    diagnostics = build_test_diagnostics(
        n_candidates_evaluated=n_evaluated,
        n_valid_candidates=n_valid,
        n_failed_candidates=0,
        warnings=warnings,
        lambdas_y=lambdas_y,
        periodogram_y=I_y,
        r_peak=r_peak,
        r_candidates=r_candidates,
        d_grid=d_grid,
        config=config,
        adaptive_info=adaptive_info,
    )

    return CyclicalFractionalTestResult(
        best_result=best_result,
        top_k_results=top_k_results,
        r_peak=r_peak,
        r_candidates=r_candidates,
        d_grid=d_grid,
        config=config,
        n_candidates_evaluated=n_evaluated,
        diagnostics=diagnostics,
        under_threshold_results=under_threshold_results,
    )


def _run_multi_cycle_fractional_test(
    *,
    arr: np.ndarray,
    X: np.ndarray,
    lambdas_y: np.ndarray,
    I_y: np.ndarray,
    periodogram: np.ndarray,
    d_grid: np.ndarray,
    config: CyclicalTestConfig,
    threshold_value: Optional[float],
    ignored_rs: set[int],
) -> CyclicalFractionalTestResult:
    """Run the scalar aggregate-ψ multi-cycle joint D search."""
    r_candidates = _find_top_periodogram_peaks_excluding(
        periodogram,
        n_peaks=config.n_stochastic_cycles,
        ignored_rs=ignored_rs,
    )
    r_peak = int(r_candidates[0])

    under_threshold_results: Optional[Dict[Tuple[int, ...], List[GridCandidateResult]]] = (
        {} if threshold_value is not None else None
    )

    n_evaluated = 0
    warnings: List[str] = []
    adaptive_info: Optional[dict] = None
    selector = TopKSelector(k=config.top_k, statistic_mode=config.statistic_mode)

    if config.d_search_strategy == "fixed_grid":
        for cycles in build_multi_cycle_candidate_grid(r_candidates, d_grid):
            candidate = evaluate_candidate(arr, X, cycles, config)
            _log_multi_cycle_candidate(candidate, config.statistic_mode)
            selector.consider(candidate)
            if under_threshold_results is not None:
                _record_if_under_threshold(
                    under_threshold_results,
                    candidate,
                    threshold_value,
                    config.statistic_mode,
                    config.return_residuals_for_threshold,
                )
            n_evaluated += 1
    else:
        results_by_d: Dict[Tuple[float, ...], GridCandidateResult] = {}
        n_coarse = 0
        n_fine = 0

        for cycles in build_multi_cycle_candidate_grid(r_candidates, d_grid):
            key = _cycles_d_key(cycles)
            if key in results_by_d:
                continue
            candidate = evaluate_candidate(arr, X, cycles, config)
            results_by_d[key] = candidate
            _log_multi_cycle_candidate(
                candidate, config.statistic_mode, label="multi-cycle coarse candidate"
            )
            selector.consider(candidate)
            if under_threshold_results is not None:
                _record_if_under_threshold(
                    under_threshold_results,
                    candidate,
                    threshold_value,
                    config.statistic_mode,
                    config.return_residuals_for_threshold,
                )
            n_coarse += 1

        best_coarse_d = min(
            results_by_d,
            key=lambda key: score_candidate(results_by_d[key], config.statistic_mode),
        )
        fine_d_grids = [
            build_d_fine_grid(
                center,
                config.d_fine_radius,
                config.d_fine_step,
            )
            for center in best_coarse_d
        ]
        for cycles in build_multi_cycle_candidate_grid_from_d_grids(
            r_candidates, fine_d_grids
        ):
            key = _cycles_d_key(cycles)
            if key in results_by_d:
                continue
            candidate = evaluate_candidate(arr, X, cycles, config)
            results_by_d[key] = candidate
            _log_multi_cycle_candidate(
                candidate, config.statistic_mode, label="multi-cycle fine candidate"
            )
            selector.consider(candidate)
            if under_threshold_results is not None:
                _record_if_under_threshold(
                    under_threshold_results,
                    candidate,
                    threshold_value,
                    config.statistic_mode,
                    config.return_residuals_for_threshold,
                )
            n_fine += 1

        best_result = selector.get_best()
        if best_result is None:
            raise InvalidConfigurationError(
                "No multi-cycle adaptive candidates were evaluated."
            )
        n_evaluated = n_coarse + n_fine
        adaptive_info = {
            "d_coarse_grid": d_grid,
            "d_fine_step": config.d_fine_step,
            "d_fine_radius": config.d_fine_radius,
            "best_coarse_d_per_r": [float(value) for value in best_coarse_d],
            "final_d_per_r": [cycle.D for cycle in best_result.cycles],
            "n_coarse_evaluations": n_coarse,
            "n_fine_evaluations": n_fine,
        }

    if under_threshold_results is not None:
        under_threshold_results = {
            R: sorted(candidates, key=lambda c: score_candidate(c, config.statistic_mode))
            for R, candidates in sorted(under_threshold_results.items())
        }

    diagnostics = build_test_diagnostics(
        n_candidates_evaluated=n_evaluated,
        n_valid_candidates=n_evaluated,
        n_failed_candidates=0,
        warnings=warnings,
        lambdas_y=lambdas_y,
        periodogram_y=I_y,
        r_peak=r_peak,
        r_candidates=r_candidates,
        d_grid=d_grid,
        config=config,
        adaptive_info=adaptive_info,
    )

    return CyclicalFractionalTestResult(
        best_result=selector.get_best(),
        top_k_results=selector.get_top_k(),
        r_peak=r_peak,
        r_candidates=r_candidates,
        d_grid=d_grid,
        config=config,
        n_candidates_evaluated=n_evaluated,
        diagnostics=diagnostics,
        under_threshold_results=under_threshold_results,
    )


def _validate_threshold(threshold: Any) -> float:
    """Return threshold as a positive finite float, or raise InvalidConfigurationError."""
    if isinstance(threshold, bool):
        raise InvalidConfigurationError("threshold must be a real number, got bool.")
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"threshold must be numeric: {exc}") from exc
    if not np.isfinite(value):
        raise InvalidConfigurationError(f"threshold must be finite, got {threshold!r}.")
    if value <= 0.0:
        raise InvalidConfigurationError(f"threshold must be > 0, got {value}.")
    return value


def _build_ignored_r_set(config: CyclicalTestConfig, T: int) -> set[int]:
    """Return stochastic frequency indices excluded from candidate search."""
    ignored_rs = set()
    if config.exclude_zero_frequency:
        ignored_rs.add(0)
    if config.ignored_stochastic_rs is not None:
        ignored_rs.update(int(value) for value in config.ignored_stochastic_rs)

    out_of_range = sorted(value for value in ignored_rs if value < 0 or value >= T)
    if out_of_range:
        raise InvalidConfigurationError(
            "ignored_stochastic_rs contains values outside the valid cycle range "
            f"[0, {T - 1}]: {out_of_range}."
        )
    return ignored_rs


def _find_periodogram_peak_excluding(
    periodogram: np.ndarray,
    ignored_rs: set[int],
) -> int:
    """Return the strongest searchable periodogram index after exclusions."""
    per = np.asarray(periodogram, dtype=float)
    available = _available_periodogram_indices(per, ignored_rs)
    return int(available[np.argmax(per[available])])


def _find_top_periodogram_peaks_excluding(
    periodogram: np.ndarray,
    *,
    n_peaks: int,
    ignored_rs: set[int],
) -> np.ndarray:
    """Return strongest searchable periodogram indices after exclusions."""
    per = np.asarray(periodogram, dtype=float)
    available = _available_periodogram_indices(per, ignored_rs)
    if len(available) < n_peaks:
        raise InvalidConfigurationError(
            f"Need {n_peaks} stochastic cycle frequencies, but only "
            f"{len(available)} remain after ignored_stochastic_rs filtering."
        )
    top_local = np.argsort(per[available])[-n_peaks:][::-1]
    return available[top_local].astype(int)


def _available_periodogram_indices(
    periodogram: np.ndarray,
    ignored_rs: set[int],
) -> np.ndarray:
    if periodogram.ndim != 1 or periodogram.size == 0:
        raise InvalidConfigurationError("periodogram must be a non-empty 1-D array.")
    if not np.all(np.isfinite(periodogram)):
        raise InvalidConfigurationError("periodogram contains non-finite values.")

    mask = np.ones(periodogram.size, dtype=bool)
    for R in ignored_rs:
        if R < periodogram.size:
            mask[R] = False
    available = np.flatnonzero(mask)
    if available.size == 0:
        raise InvalidConfigurationError(
            "No periodogram frequencies remain after ignored_stochastic_rs filtering."
        )
    return available


def _filter_ignored_r_candidates(
    r_candidates: np.ndarray,
    ignored_rs: set[int],
) -> np.ndarray:
    filtered = np.array(
        [int(R) for R in np.asarray(r_candidates, dtype=int) if int(R) not in ignored_rs],
        dtype=int,
    )
    if filtered.size == 0:
        raise InvalidConfigurationError(
            "No stochastic R candidates remain after ignored_stochastic_rs filtering."
        )
    return filtered


def _record_if_under_threshold(
    bucket: Dict[Tuple[int, ...], List[GridCandidateResult]],
    candidate: GridCandidateResult,
    threshold: float,
    statistic_mode: str,
    retain_residuals: bool,
) -> None:
    """Append candidate to its full R-tuple bucket when its score is below threshold."""
    if score_candidate(candidate, statistic_mode) < threshold:
        key = tuple(int(cycle.R) for cycle in candidate.cycles)
        bucket.setdefault(key, []).append(
            candidate
            if retain_residuals
            else dataclasses.replace(candidate, residuals=None)
        )


def _log_multi_cycle_candidate(
    candidate: GridCandidateResult,
    statistic_mode: str,
    *,
    label: str = "multi-cycle candidate",
) -> None:
    """Log the statistic values for one evaluated multi-cycle candidate."""
    logger.info(
        "%s cycles=%s TEST=%.6g TEST*=%.6g score(%s)=%.6g XA=%.6g XAA=%.6g",
        label,
        _format_cycles_for_log(candidate.cycles),
        candidate.test_value,
        candidate.test_star_value,
        statistic_mode,
        score_candidate(candidate, statistic_mode),
        candidate.xa,
        candidate.xaa,
    )


def _format_cycles_for_log(cycles: Tuple[StochasticCycle, ...]) -> str:
    """Return a compact cycles string for logging."""
    return "[" + ", ".join(
        f"(R={cycle.R},D={cycle.D:.4f})" for cycle in cycles
    ) + "]"


def _cycles_d_key(cycles: Tuple[StochasticCycle, ...]) -> Tuple[float, ...]:
    """Return the rounded D-vector key used to de-duplicate adaptive candidates."""
    return tuple(round(float(cycle.D), 12) for cycle in cycles)
