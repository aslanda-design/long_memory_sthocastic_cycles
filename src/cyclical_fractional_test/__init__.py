from importlib.metadata import PackageNotFoundError, version

from .api import (
    compute_autocorrelogram,
    compute_periodogram,
    run_cyclical_fractional_test,
)
from .diagnostics import (
    PeriodogramSummary,
    TestDiagnostics,
    VarianceComparison,
    build_candidate_diagnostics,
    build_test_diagnostics,
    compare_variance_definitions,
    summarize_periodogram,
)
from .evaluation import evaluate_candidate, evaluate_r_with_adaptive_d
from .chebyshev import (
    build_chebyshev_design,
    build_chebyshev_design_at,
    build_single_chebyshev_polynomial,
    evaluate_single_chebyshev_polynomial,
)
from .config import CyclicalTestConfig
from .exceptions import (
    CyclicalFractionalTestError,
    InvalidCycleError,
    InvalidConfigurationError,
    InvalidSeriesError,
    NotFittedError,
)
from .model import CyclicalFractionalModel
from .prediction import (
    compute_ma_weights,
    forecast_ar,
    forecast_out_of_sample,
    reconstruct_in_sample,
)
from .grid import (
    build_d_fine_grid,
    build_d_grid,
    build_d_grid_for_strategy,
    build_default_d_coarse_grid,
    build_multi_cycle_candidate_grid,
    build_multi_cycle_candidate_grid_from_d_grids,
    build_r_grid_around_peak,
    build_single_cycle_candidate_grid,
    candidate_iterator,
)
from .results import (
    AdaptiveDSearchResult,
    CyclicalFractionalTestResult,
    GridCandidateResult,
    StochasticCycle,
)
from .filters import (
    apply_filter_dynamic,
    apply_fractional_filter_single_series,
    apply_multi_cycle_filter,
    apply_single_cycle_filter,
    compute_fractional_coefficients_dynamic,
    compute_fractional_coefficients_from_mu,
    compute_fractional_coefficients_multi_cycle,
    compute_fractional_coefficients_single_cycle,
    compute_mu,
    filter_response_and_design,
)
from .regression import (
    BetaSignificanceResult,
    DEFAULT_BETA_SIGNIFICANCE_CRITICAL_VALUE,
    RegressionResult,
    compute_beta_standard_errors,
    compute_beta_t_statistics,
    compute_residual_sum_squares,
    compute_residuals,
    compute_time_variance,
    detect_beta_significance,
    estimate_ar_ols,
    estimate_innovation_variance,
    fit_filtered_regression,
)
from .scoring import (
    TopKSelector,
    compute_test_star_statistic,
    compute_test_statistic,
    score_candidate,
)
from .spectral import (
    compute_ar_spectral_adjustment,
    compute_document_periodogram,
    compute_frequency_variance_dynamic,
    compute_frequency_variance_multi_cycle,
    compute_frequency_variance_single_cycle,
    compute_psi_dynamic,
    compute_psi_multi_cycle,
    compute_psi_single_cycle,
    compute_residual_periodogram,
    compute_xa_ar_adjusted,
    compute_xa_ar1_dynamic,
    compute_xa_ar1_multi_cycle,
    compute_xa_ar1_single_cycle,
    compute_xa_ar2_dynamic,
    compute_xa_ar2_multi_cycle,
    compute_xa_ar2_single_cycle,
    compute_xa_dynamic,
    compute_xa_error_model,
    compute_xa_multi_cycle,
    compute_xa_single_cycle,
    compute_xaa_dynamic,
    compute_xaa_ar1_dynamic,
    compute_xaa_ar1_multi_cycle,
    compute_xaa_ar1_single_cycle,
    compute_xaa_ar2_dynamic,
    compute_xaa_ar2_multi_cycle,
    compute_xaa_ar2_single_cycle,
    compute_xaa_error_model,
    compute_xaa_multi_cycle,
    compute_xaa_single_cycle,
    find_periodogram_peak,
    find_top_periodogram_peaks,
)

try:
    __version__ = version("cyclical-fractional-test")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    # api
    "compute_autocorrelogram",
    "compute_periodogram",
    "run_cyclical_fractional_test",
    # evaluation — Wave 12
    "evaluate_candidate",
    # evaluation — adaptive D search
    "evaluate_r_with_adaptive_d",
    "AdaptiveDSearchResult",
    # diagnostics — Wave 14
    "PeriodogramSummary",
    "VarianceComparison",
    "TestDiagnostics",
    "summarize_periodogram",
    "compare_variance_definitions",
    "build_candidate_diagnostics",
    "build_test_diagnostics",
    # config
    "CyclicalTestConfig",
    # model — sklearn-style estimator
    "CyclicalFractionalModel",
    # prediction
    "reconstruct_in_sample",
    "forecast_ar",
    "forecast_out_of_sample",
    "compute_ma_weights",
    # results
    "StochasticCycle",
    "GridCandidateResult",
    "CyclicalFractionalTestResult",
    # exceptions
    "CyclicalFractionalTestError",
    "InvalidSeriesError",
    "InvalidConfigurationError",
    "InvalidCycleError",
    "NotFittedError",
    # chebyshev
    "build_single_chebyshev_polynomial",
    "evaluate_single_chebyshev_polynomial",
    "build_chebyshev_design",
    "build_chebyshev_design_at",
    # spectral — periodogram
    "compute_document_periodogram",
    "find_periodogram_peak",
    "find_top_periodogram_peaks",
    # spectral — psi / XAA
    "compute_psi_single_cycle",
    "compute_psi_multi_cycle",
    "compute_xaa_single_cycle",
    "compute_xaa_multi_cycle",
    "compute_psi_dynamic",
    "compute_xaa_dynamic",
    "compute_xaa_error_model",
    "compute_xaa_ar1_single_cycle",
    "compute_xaa_ar1_multi_cycle",
    "compute_xaa_ar1_dynamic",
    "compute_xaa_ar2_single_cycle",
    "compute_xaa_ar2_multi_cycle",
    "compute_xaa_ar2_dynamic",
    # grid
    "build_r_grid_around_peak",
    "build_d_grid",
    "build_default_d_coarse_grid",
    "build_d_fine_grid",
    "build_d_grid_for_strategy",
    "build_single_cycle_candidate_grid",
    "build_multi_cycle_candidate_grid",
    "build_multi_cycle_candidate_grid_from_d_grids",
    "candidate_iterator",
    # filters — Wave 6
    "compute_mu",
    "compute_fractional_coefficients_single_cycle",
    "compute_fractional_coefficients_multi_cycle",
    "compute_fractional_coefficients_dynamic",
    # filters — Wave 7
    "apply_fractional_filter_single_series",
    "apply_single_cycle_filter",
    "apply_multi_cycle_filter",
    "apply_filter_dynamic",
    "filter_response_and_design",
    # regression — Wave 8
    "BetaSignificanceResult",
    "DEFAULT_BETA_SIGNIFICANCE_CRITICAL_VALUE",
    "RegressionResult",
    "fit_filtered_regression",
    "compute_beta_standard_errors",
    "compute_beta_t_statistics",
    "detect_beta_significance",
    "compute_residuals",
    "compute_residual_sum_squares",
    "compute_time_variance",
    "estimate_ar_ols",
    "estimate_innovation_variance",
    # filters — coefficient builder
    "compute_fractional_coefficients_from_mu",
    # spectral — Wave 9
    "compute_ar_spectral_adjustment",
    "compute_residual_periodogram",
    "compute_frequency_variance_single_cycle",
    "compute_frequency_variance_multi_cycle",
    "compute_frequency_variance_dynamic",
    # spectral — Wave 10
    "compute_xa_single_cycle",
    "compute_xa_multi_cycle",
    "compute_xa_dynamic",
    "compute_xa_ar_adjusted",
    "compute_xa_error_model",
    "compute_xa_ar1_single_cycle",
    "compute_xa_ar1_multi_cycle",
    "compute_xa_ar1_dynamic",
    "compute_xa_ar2_single_cycle",
    "compute_xa_ar2_multi_cycle",
    "compute_xa_ar2_dynamic",
    # scoring — Wave 11
    "compute_test_statistic",
    "compute_test_star_statistic",
    "score_candidate",
    "TopKSelector",
]
