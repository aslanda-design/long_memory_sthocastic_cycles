import numpy as np
import pytest

from cyclical_fractional_test.config import CyclicalTestConfig
from cyclical_fractional_test.diagnostics import (
    PeriodogramSummary,
    TestDiagnostics,
    VarianceComparison,
    build_candidate_diagnostics,
    build_test_diagnostics,
    compare_variance_definitions,
    summarize_periodogram,
)
from cyclical_fractional_test.exceptions import InvalidConfigurationError
from cyclical_fractional_test.results import GridCandidateResult, StochasticCycle


# ---------------------------------------------------------------------------
# summarize_periodogram
# ---------------------------------------------------------------------------


def _make_periodogram(T=8):
    rng = np.random.default_rng(0)
    per = rng.random(T)
    lam = 2 * np.pi * np.arange(T) / T
    return lam, per


def test_summarize_periodogram_returns_summary():
    lam, per = _make_periodogram()
    result = summarize_periodogram(lam, per)
    assert isinstance(result, PeriodogramSummary)


def test_summarize_periodogram_known_peak():
    I = np.array([0.1, 0.5, 3.0, 0.3, 0.8])
    lam = 2 * np.pi * np.arange(5) / 5
    result = summarize_periodogram(lam, I, peak_index=2, exclude_zero=False)
    assert result.peak_index == 2
    assert np.isclose(result.peak_value, 3.0)


def test_summarize_periodogram_computes_peak_when_not_given():
    I = np.array([10.0, 0.5, 2.0, 0.3, 0.8])
    lam = 2 * np.pi * np.arange(5) / 5
    result = summarize_periodogram(lam, I, exclude_zero=True)
    # index 0 (value 10.0) is excluded; next max is index 2 (value 2.0)
    assert result.peak_index == 2
    assert result.exclude_zero_used is True


def test_summarize_periodogram_exclude_zero_false_picks_zero():
    I = np.array([10.0, 0.5, 2.0])
    lam = 2 * np.pi * np.arange(3) / 3
    result = summarize_periodogram(lam, I, exclude_zero=False)
    assert result.peak_index == 0


def test_summarize_periodogram_top_peaks_sorted_descending():
    I = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    lam = 2 * np.pi * np.arange(5) / 5
    result = summarize_periodogram(lam, I, n_top_peaks=3, exclude_zero=False)
    values = [v for _, _, v in result.top_peaks]
    assert values == sorted(values, reverse=True)


def test_summarize_periodogram_total_power():
    I = np.array([1.0, 2.0, 3.0, 4.0])
    lam = np.zeros(4)
    result = summarize_periodogram(lam, I, exclude_zero=False)
    assert np.isclose(result.total_power, 10.0)


def test_summarize_periodogram_n_top_peaks_capped_by_available():
    I = np.array([1.0, 2.0, 3.0])
    lam = np.zeros(3)
    result = summarize_periodogram(lam, I, n_top_peaks=10, exclude_zero=False)
    assert len(result.top_peaks) == 3


def test_summarize_periodogram_rejects_mismatched_lengths():
    lam = np.arange(5) * 0.1
    per = np.arange(4) * 0.1
    with pytest.raises((InvalidConfigurationError, ValueError)):
        summarize_periodogram(lam, per)


def test_summarize_periodogram_rejects_empty_arrays():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        summarize_periodogram(np.array([]), np.array([]))


def test_summarize_periodogram_rejects_invalid_n_top_peaks():
    lam, per = _make_periodogram()
    with pytest.raises((InvalidConfigurationError, ValueError)):
        summarize_periodogram(lam, per, n_top_peaks=0)


# ---------------------------------------------------------------------------
# compare_variance_definitions
# ---------------------------------------------------------------------------


def test_compare_variance_equal_values():
    result = compare_variance_definitions(2.0, 2.0)
    assert isinstance(result, VarianceComparison)
    assert result.absolute_difference == 0.0
    assert result.relative_difference == 0.0


def test_compare_variance_known_values():
    result = compare_variance_definitions(3.0, 1.0)
    assert np.isclose(result.absolute_difference, 2.0)
    assert np.isclose(result.relative_difference, 2.0 / 3.0)


def test_compare_variance_fields_match_inputs():
    result = compare_variance_definitions(0.5, 0.3)
    assert np.isclose(result.variance_time, 0.5)
    assert np.isclose(result.variance_frequency, 0.3)


def test_compare_variance_zero_variance_time_no_division_error():
    result = compare_variance_definitions(0.0, 1.0)
    assert np.isfinite(result.relative_difference)
    assert result.relative_difference > 0.0


def test_compare_variance_rejects_nan():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compare_variance_definitions(float("nan"), 1.0)


def test_compare_variance_rejects_inf():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compare_variance_definitions(1.0, float("inf"))


# ---------------------------------------------------------------------------
# build_candidate_diagnostics
# ---------------------------------------------------------------------------


def _make_candidate(**kwargs):
    defaults = dict(
        cycles=(StochasticCycle(R=2, D=0.3),),
        test_value=-0.5,
        test_star_value=0.7,
        variance_time=0.3,
        variance_frequency=0.4,
        betas=np.array([1.0, 2.0]),
        residuals=np.ones(10),
    )
    defaults.update(kwargs)
    return GridCandidateResult(**defaults)


def test_build_candidate_diagnostics_returns_dict():
    result = build_candidate_diagnostics(_make_candidate())
    assert isinstance(result, dict)


def test_build_candidate_diagnostics_has_finite_test():
    result = build_candidate_diagnostics(_make_candidate(test_value=-0.5))
    assert result["has_finite_test"] is True


def test_build_candidate_diagnostics_has_finite_test_star():
    result = build_candidate_diagnostics(_make_candidate(test_star_value=0.7))
    assert result["has_finite_test_star"] is True


def test_build_candidate_diagnostics_residual_variance_positive():
    result = build_candidate_diagnostics(_make_candidate(variance_time=0.3))
    assert result["residual_variance_positive"] is True


def test_build_candidate_diagnostics_number_of_betas():
    result = build_candidate_diagnostics(_make_candidate(betas=np.array([1.0, 2.0, 3.0])))
    assert result["number_of_betas"] == 3


def test_build_candidate_diagnostics_residual_length():
    result = build_candidate_diagnostics(_make_candidate(residuals=np.ones(15)))
    assert result["residual_length"] == 15


def test_build_candidate_diagnostics_variance_comparison_present():
    result = build_candidate_diagnostics(_make_candidate(variance_time=0.3, variance_frequency=0.4))
    assert result["variance_comparison"] is not None
    assert isinstance(result["variance_comparison"], VarianceComparison)


def test_build_candidate_diagnostics_includes_error_model_metadata():
    result = build_candidate_diagnostics(
        _make_candidate(error_model="ar1", ar_coefficients=(0.4,))
    )
    assert result["error_model"] == "ar1"
    assert result["ar_coefficients"] == (0.4,)


# ---------------------------------------------------------------------------
# build_test_diagnostics
# ---------------------------------------------------------------------------


def test_build_test_diagnostics_basic_counters():
    result = build_test_diagnostics(
        n_candidates_evaluated=10,
        n_valid_candidates=10,
        n_failed_candidates=0,
        warnings=[],
        lambdas_y=None,
        periodogram_y=None,
        r_peak=3,
        r_candidates=np.array([2, 3, 4]),
        d_grid=np.array([0.0, 0.5]),
        config=CyclicalTestConfig(
            stochastic_cycle_mode="multi_cycle", n_stochastic_cycles=3
        ),
    )
    assert isinstance(result, TestDiagnostics)
    assert result.n_candidates_evaluated == 10
    assert result.n_valid_candidates == 10
    assert result.n_failed_candidates == 0
    assert result.r_peak == 3
    assert result.r_candidates_count == 3
    assert result.d_grid_count == 2
    assert result.n_stochastic_cycles == 3


def test_build_test_diagnostics_no_periodogram_returns_none_summary():
    result = build_test_diagnostics(
        n_candidates_evaluated=5,
        n_valid_candidates=5,
        n_failed_candidates=0,
        warnings=[],
        lambdas_y=None,
        periodogram_y=None,
        r_peak=2,
        r_candidates=np.array([1, 2]),
        d_grid=np.array([0.0]),
        config=CyclicalTestConfig(),
    )
    assert result.periodogram_summary is None


def test_build_test_diagnostics_with_periodogram():
    T = 12
    lam = 2 * np.pi * np.arange(T) / T
    per = np.random.default_rng(0).random(T)
    result = build_test_diagnostics(
        n_candidates_evaluated=4,
        n_valid_candidates=4,
        n_failed_candidates=0,
        warnings=[],
        lambdas_y=lam,
        periodogram_y=per,
        r_peak=2,
        r_candidates=np.array([1, 2, 3]),
        d_grid=np.array([0.0, 0.5]),
        config=CyclicalTestConfig(),
    )
    assert result.periodogram_summary is not None
    assert result.periodogram_summary.peak_index == 2


def test_build_test_diagnostics_warnings_copied():
    result = build_test_diagnostics(
        n_candidates_evaluated=0,
        n_valid_candidates=0,
        n_failed_candidates=0,
        warnings=["w1", "w2"],
        lambdas_y=None,
        periodogram_y=None,
        r_peak=None,
        r_candidates=np.array([]),
        d_grid=np.array([]),
        config=CyclicalTestConfig(),
    )
    assert result.warnings == ["w1", "w2"]


def test_build_test_diagnostics_statistic_mode_from_config():
    config = CyclicalTestConfig(statistic_mode="test_star")
    result = build_test_diagnostics(
        n_candidates_evaluated=1,
        n_valid_candidates=1,
        n_failed_candidates=0,
        warnings=[],
        lambdas_y=None,
        periodogram_y=None,
        r_peak=1,
        r_candidates=np.array([1]),
        d_grid=np.array([0.0]),
        config=config,
    )
    assert result.selected_statistic_mode == "test_star"


def test_build_test_diagnostics_error_model_from_config():
    result = build_test_diagnostics(
        n_candidates_evaluated=1,
        n_valid_candidates=1,
        n_failed_candidates=0,
        warnings=[],
        lambdas_y=None,
        periodogram_y=None,
        r_peak=1,
        r_candidates=np.array([1]),
        d_grid=np.array([0.0]),
        config=CyclicalTestConfig(error_model="ar2"),
    )
    assert result.error_model == "ar2"
