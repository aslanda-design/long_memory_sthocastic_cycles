from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .exceptions import InvalidConfigurationError


@dataclass
class PeriodogramSummary:
    """Summary statistics of the input series periodogram."""

    peak_index: int  # R* — argmax of periodogram (zero excluded by default).
    peak_frequency: float  # λ_{R*} = 2π R* / T.
    peak_value: float  # I(λ_{R*}).
    top_peaks: List[Tuple[int, float, float]]  # (index, frequency, value) sorted descending.
    total_power: float  # Σ_j I(λ_j).
    exclude_zero_used: bool  # Whether j=0 was skipped when locating the peak.


@dataclass
class VarianceComparison:
    """Side-by-side comparison of the two variance estimators VAR and VAR*.

    Both should be close when the model is correctly specified.
    """

    variance_time: float  # VAR — (1/T) Σ_t residuals(t)².
    variance_frequency: float  # VAR* — (2π/T) Σ_j I_residuals(λ_j).
    absolute_difference: float  # |VAR − VAR*|.
    relative_difference: float  # |VAR − VAR*| / max(|VAR|, ε).


@dataclass
class TestDiagnostics:
    """Run-level diagnostic summary attached to CyclicalFractionalTestResult."""

    __test__ = False

    n_candidates_evaluated: int  # Total candidates passed to evaluate_candidate.
    n_valid_candidates: int  # Candidates that returned a finite result.
    n_failed_candidates: int  # Candidates that raised an error (0 under Policy A).
    warnings: List[str]  # Warning messages accumulated during the run.
    periodogram_summary: Optional[PeriodogramSummary]  # Input periodogram info.
    selected_statistic_mode: str  # config.statistic_mode used for ranking.
    stochastic_cycle_mode: str  # config.stochastic_cycle_mode used.
    n_stochastic_cycles: int  # Number of stochastic cycles requested.
    error_model: str  # config.error_model used for residual errors.
    r_peak: Optional[int]  # Main periodogram peak index.
    r_candidates_count: int  # Number of R values in the search grid.
    d_grid_count: int  # Number of D values in the (coarse or fixed) search grid.
    d_search_strategy: str = "adaptive"  # "adaptive" or "fixed_grid".
    d_fine_step: Optional[float] = None  # Fine-grid step (adaptive search only).
    d_fine_radius: Optional[float] = None  # Fine-grid half-width (adaptive search only).
    best_coarse_d_per_r: Optional[List[float]] = None  # Best coarse D values in R order (adaptive).
    final_d_per_r: Optional[List[float]] = None  # Final refined D values in R order (adaptive).
    n_coarse_evaluations: Optional[int] = None  # Coarse candidates evaluated (adaptive).
    n_fine_evaluations: Optional[int] = None  # Fine candidates evaluated (adaptive).


def summarize_periodogram(
    lambdas: np.ndarray,
    periodogram: np.ndarray,
    peak_index: Optional[int] = None,
    n_top_peaks: int = 5,
    exclude_zero: bool = True,
) -> PeriodogramSummary:
    """Summarise the periodogram: dominant frequency, top-n peaks, total power.

    If peak_index is None it is recomputed (skipping j=0 when exclude_zero=True).
    Top peaks are sorted largest first.
    """
    _validate_summarize_periodogram(lambdas, periodogram, n_top_peaks)
    lam = np.asarray(lambdas, dtype=float)
    per = np.asarray(periodogram, dtype=float)
    T = len(per)

    if peak_index is None:
        search = per[1:] if exclude_zero else per
        offset = 1 if exclude_zero else 0
        peak_index = int(np.argmax(search)) + offset

    peak_frequency = float(lam[peak_index])
    peak_value = float(per[peak_index])
    total_power = float(np.sum(per))

    candidates = per[1:] if exclude_zero else per
    offset = 1 if exclude_zero else 0
    n_avail = min(n_top_peaks, len(candidates))
    top_idx = np.argsort(candidates)[-n_avail:][::-1] + offset
    top_peaks = [(int(i), float(lam[i]), float(per[i])) for i in top_idx]

    return PeriodogramSummary(
        peak_index=peak_index,
        peak_frequency=peak_frequency,
        peak_value=peak_value,
        top_peaks=top_peaks,
        total_power=total_power,
        exclude_zero_used=exclude_zero,
    )


def compare_variance_definitions(
    variance_time: float,
    variance_frequency: float,
) -> VarianceComparison:
    """Compare VAR (time-domain) and VAR* (frequency-domain).

    absolute_difference = |VAR − VAR*|
    relative_difference = absolute_difference / max(|VAR|, ε)

    variance_time = 0 is handled without division by zero via ε = 1e-12.
    """
    _validate_compare_variance_definitions(variance_time, variance_frequency)
    vt = float(variance_time)
    vf = float(variance_frequency)
    abs_diff = abs(vt - vf)
    rel_diff = abs_diff / max(abs(vt), 1e-12)
    return VarianceComparison(
        variance_time=vt,
        variance_frequency=vf,
        absolute_difference=abs_diff,
        relative_difference=rel_diff,
    )


def build_candidate_diagnostics(candidate_result: Any) -> dict:
    """Build a diagnostic dict for one GridCandidateResult.

    Returns a plain dict so callers can log or inspect without imports.
    """
    vt = candidate_result.variance_time
    vf = candidate_result.variance_frequency
    test_val = candidate_result.test_value
    test_star_val = candidate_result.test_star_value
    betas = candidate_result.betas
    residuals = candidate_result.residuals

    variance_comparison = None
    if vt is not None and vf is not None:
        try:
            variance_comparison = compare_variance_definitions(vt, vf)
        except Exception:
            pass

    return {
        "variance_comparison": variance_comparison,
        "has_finite_test": (
            bool(np.isfinite(test_val)) if test_val is not None else False
        ),
        "has_finite_test_star": (
            bool(np.isfinite(test_star_val)) if test_star_val is not None else False
        ),
        "residual_variance_positive": (
            bool(vt > 0) if vt is not None else False
        ),
        "number_of_betas": len(betas) if betas is not None else None,
        "residual_length": len(residuals) if residuals is not None else None,
        "error_model": getattr(candidate_result, "error_model", "white_noise"),
        "ar_coefficients": getattr(candidate_result, "ar_coefficients", ()),
    }


def build_test_diagnostics(
    *,
    n_candidates_evaluated: int,
    n_valid_candidates: int,
    n_failed_candidates: int,
    warnings: List[str],
    lambdas_y: Optional[np.ndarray],
    periodogram_y: Optional[np.ndarray],
    r_peak: Optional[int],
    r_candidates: Any,
    d_grid: Any,
    config: Any,
    adaptive_info: Optional[dict] = None,
) -> TestDiagnostics:
    """Assemble the run-level TestDiagnostics from collected counters and arrays.

    adaptive_info, when present, carries the per-R coarse/fine search summary; it
    is purely descriptive and never feeds back into the statistical computation.
    """
    periodogram_summary = None
    if lambdas_y is not None and periodogram_y is not None:
        try:
            periodogram_summary = summarize_periodogram(
                lambdas_y,
                periodogram_y,
                peak_index=r_peak,
                exclude_zero=getattr(config, "exclude_zero_frequency", True),
            )
        except Exception:
            pass

    r_count = len(r_candidates) if r_candidates is not None else 0
    d_count = len(d_grid) if d_grid is not None else 0
    info = adaptive_info or {}

    return TestDiagnostics(
        n_candidates_evaluated=n_candidates_evaluated,
        n_valid_candidates=n_valid_candidates,
        n_failed_candidates=n_failed_candidates,
        warnings=list(warnings),
        periodogram_summary=periodogram_summary,
        selected_statistic_mode=getattr(config, "statistic_mode", "unknown"),
        stochastic_cycle_mode=getattr(config, "stochastic_cycle_mode", "unknown"),
        n_stochastic_cycles=getattr(config, "n_stochastic_cycles", 1),
        error_model=getattr(config, "error_model", "unknown"),
        r_peak=r_peak,
        r_candidates_count=r_count,
        d_grid_count=d_count,
        d_search_strategy=getattr(config, "d_search_strategy", "adaptive"),
        d_fine_step=info.get("d_fine_step"),
        d_fine_radius=info.get("d_fine_radius"),
        best_coarse_d_per_r=info.get("best_coarse_d_per_r"),
        final_d_per_r=info.get("final_d_per_r"),
        n_coarse_evaluations=info.get("n_coarse_evaluations"),
        n_fine_evaluations=info.get("n_fine_evaluations"),
    )


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know the exact
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_summarize_periodogram(
    lambdas: Any, periodogram: Any, n_top_peaks: int
) -> None:
    try:
        lam = np.asarray(lambdas, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"lambdas must be numeric: {exc}") from exc
    try:
        per = np.asarray(periodogram, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"periodogram must be numeric: {exc}") from exc
    if lam.ndim != 1 or lam.size == 0:
        raise InvalidConfigurationError("lambdas must be a non-empty 1-D array.")
    if per.ndim != 1 or per.size == 0:
        raise InvalidConfigurationError("periodogram must be a non-empty 1-D array.")
    if lam.shape != per.shape:
        raise InvalidConfigurationError(
            f"lambdas and periodogram must have the same length; "
            f"got {lam.shape} and {per.shape}."
        )
    if isinstance(n_top_peaks, bool) or not isinstance(n_top_peaks, int) or n_top_peaks < 1:
        raise InvalidConfigurationError(
            f"n_top_peaks must be a positive int, got {n_top_peaks!r}."
        )


def _validate_compare_variance_definitions(
    variance_time: Any, variance_frequency: Any
) -> None:
    for val, name in [(variance_time, "variance_time"), (variance_frequency, "variance_frequency")]:
        try:
            v = float(val)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(f"{name} must be numeric: {exc}") from exc
        if not np.isfinite(v):
            raise InvalidConfigurationError(
                f"{name} must be finite, got {val!r}."
            )
