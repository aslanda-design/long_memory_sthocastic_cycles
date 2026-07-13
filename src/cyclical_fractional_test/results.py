from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CyclicalTestConfig


@dataclass
class StochasticCycle:
    """Candidate stochastic cycle used by the test.

    It represents the factor (1 - 2cos(2πR/T)L + L²)^D applied to the series.
    """

    R: int  # Candidate index for the cyclic frequency.
    D: float  # Fractional integration parameter for this cycle.


@dataclass
class GridCandidateResult:
    """Values obtained when evaluating one candidate from the grid.

    Most entries are optional while the numerical core is still being filled in.
    """

    cycles: Tuple[StochasticCycle, ...]  # Cycle tuple represented by this grid point.
    error_model: str = "white_noise"  # Residual error specification used for this candidate.
    ar_coefficients: Tuple[float, ...] = field(default_factory=tuple)  # Estimated AR nuisance parameters.
    test_value: Optional[float] = None  # TEST statistic for the candidate.
    test_star_value: Optional[float] = None  # TEST* statistic for the candidate.
    abs_test_value: Optional[float] = None  # Absolute value used for TEST ranking.
    abs_test_star_value: Optional[float] = None  # Absolute value for TEST* ranking.
    xa: Optional[float] = None  # XA(R,D) scalar used in the statistic.
    xaa: Optional[float] = None  # XAA(R) scalar used in the statistic.
    variance_time: Optional[float] = None  # Time-domain variance estimate.
    variance_frequency: Optional[float] = None  # Frequency-domain variance estimate.
    betas: Optional[np.ndarray] = None  # Estimated deterministic-cycle coefficients.
    residuals: Optional[np.ndarray] = None  # Regression residuals for this candidate.
    residual_sum_squares: Optional[float] = None  # Sum of squared residuals.
    beta_standard_errors: Optional[np.ndarray] = None  # Standard errors for deterministic beta coefficients.


@dataclass
class AdaptiveDSearchResult:
    """Outcome of the adaptive coarse-to-fine D search for a single frequency R."""

    R: int  # Frequency index the search was run for.
    best_result: GridCandidateResult  # Best candidate over coarse and fine stages.
    best_coarse_result: GridCandidateResult  # Best candidate from the coarse stage.
    best_coarse_d: float  # D selected by the coarse stage.
    best_d: float  # Final D after local refinement.
    n_coarse_evaluated: int  # Distinct coarse D values evaluated.
    n_fine_evaluated: int  # Distinct fine D values evaluated (excludes reused ones).
    n_candidates_evaluated: int  # Total distinct (R,D) candidates evaluated for this R.
    all_results: List[GridCandidateResult] = field(default_factory=list)  # Every distinct (R,D) candidate evaluated for this R.


@dataclass
class CyclicalFractionalTestResult:
    """Container returned by the cyclical fractional long-memory test."""

    best_result: Optional[GridCandidateResult] = None  # Best candidate found.
    top_k_results: List[GridCandidateResult] = field(default_factory=list)  # Retained top-k candidates.
    under_threshold_results: Optional[Dict[Tuple[int, ...], List[GridCandidateResult]]] = None  # Candidates scoring below the requested threshold, grouped by full R tuple; None when no threshold was requested.
    r_peak: Optional[int] = None  # Main periodogram peak used to build the R grid.
    r_candidates: Optional[np.ndarray] = None  # R values considered around r_peak.
    d_grid: Optional[np.ndarray] = None  # D values evaluated in the grid.
    config: Optional[CyclicalTestConfig] = None  # Configuration used in the run.
    n_candidates_evaluated: Optional[int] = None  # Total grid points evaluated.
    diagnostics: Optional[Any] = None  # TestDiagnostics; populated by run_cyclical_fractional_test.
