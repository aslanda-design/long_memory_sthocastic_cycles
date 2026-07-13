import numpy as np
import pytest

from cyclical_fractional_test import (
    CyclicalFractionalTestResult,
    CyclicalTestConfig,
    GridCandidateResult,
    StochasticCycle,
    compute_autocorrelogram,
    detect_beta_significance,
    run_cyclical_fractional_test,
)
from cyclical_fractional_test.diagnostics import PeriodogramSummary, TestDiagnostics
from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidSeriesError


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_stochastic_cycle_fields():
    c = StochasticCycle(R=25, D=0.4)
    assert c.R == 25 and c.D == 0.4


def test_grid_candidate_result_defaults_to_none():
    result = GridCandidateResult(cycles=(StochasticCycle(R=10, D=0.3),))
    assert result.test_value is None
    assert result.betas is None
    assert result.beta_standard_errors is None


def test_cyclical_result_empty_defaults():
    result = CyclicalFractionalTestResult()
    assert result.best_result is None
    assert result.top_k_results == []
    assert result.config is None


def test_config_default_values():
    cfg = CyclicalTestConfig()
    assert cfg.n_deterministic_cycles == 4
    assert cfg.chebyshev_orders is None
    assert cfg.n_stochastic_cycles == 1
    assert cfg.ignored_stochastic_rs is None
    assert cfg.top_k == 1
    assert cfg.stochastic_cycle_mode == "single"
    assert cfg.error_model == "white_noise"
    assert cfg.d_grid is None
    assert cfg.drop_singular_frequency is True
    assert cfg.return_residuals_for_threshold is False


def test_config_custom_values():
    cfg = CyclicalTestConfig(n_deterministic_cycles=2, top_k=3, r_window=5, include_intercept=True)
    assert cfg.n_deterministic_cycles == 2
    assert cfg.top_k == 3
    assert cfg.r_window == 5
    assert cfg.include_intercept is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(**overrides):
    defaults = dict(
        n_deterministic_cycles=2,
        d_grid=np.array([0.0, 0.5]),
        r_window=1,
        top_k=2,
        stochastic_cycle_mode="single",
    )
    defaults.update(overrides)
    return CyclicalTestConfig(**defaults)


def _series(T=50, seed=0):
    return np.random.default_rng(seed).standard_normal(T)


def _zero_dominant_series(T=60):
    return np.ones(T)


def _cyclic_series(T=80, components=((4, 2.0), (7, 1.0), (11, 0.8))):
    t = np.arange(T, dtype=float)
    y = np.zeros(T, dtype=float)
    for R, amplitude in components:
        y += amplitude * np.cos(2.0 * np.pi * R * t / T)
    return y


def _fixed_config(**overrides):
    defaults = dict(
        n_deterministic_cycles=2,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.5, 1.0]),
        r_window=1,
        top_k=1,
        stochastic_cycle_mode="single",
    )
    defaults.update(overrides)
    return CyclicalTestConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


def test_returns_cyclical_fractional_test_result():
    assert isinstance(
        run_cyclical_fractional_test(_series(), config=_small_config()),
        CyclicalFractionalTestResult,
    )


def test_compute_autocorrelogram_api_returns_lags_and_values():
    lags, autocorr = compute_autocorrelogram(np.array([1.0, 2.0, 3.0]), max_lag=1)
    np.testing.assert_array_equal(lags, np.array([0, 1]))
    assert autocorr[0] == pytest.approx(1.0)


def test_best_result_not_none():
    assert run_cyclical_fractional_test(_series(), config=_small_config()).best_result is not None


def test_top_k_results_length():
    result = run_cyclical_fractional_test(_series(), config=_small_config(top_k=2))
    assert len(result.top_k_results) == 2


def test_best_result_equals_first_top_k():
    result = run_cyclical_fractional_test(_series(), config=_small_config(top_k=3))
    assert result.best_result is result.top_k_results[0]


def test_betas_length_matches_n_deterministic_cycles():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=3,
            include_intercept=False,
            d_grid=np.array([0.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert all(len(c.betas) == 3 for c in result.top_k_results)
    assert all(c.beta_standard_errors.shape == (3,) for c in result.top_k_results)


def test_explicit_chebyshev_orders_control_beta_length():
    result = run_cyclical_fractional_test(
        _series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=99,
            chebyshev_orders=(2, 5),
            include_intercept=True,
            d_grid=np.array([0.0, 0.5]),
            r_window=1,
            top_k=2,
            stochastic_cycle_mode="single",
        ),
    )

    assert all(candidate.betas.shape == (3,) for candidate in result.top_k_results)
    assert all(
        candidate.beta_standard_errors.shape == (3,)
        for candidate in result.top_k_results
    )


def test_include_intercept_adds_extra_beta():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=3,
            include_intercept=True,
            d_grid=np.array([0.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert all(len(c.betas) == 4 for c in result.top_k_results)


def test_zero_deterministic_cycles_with_intercept_fits_only_intercept():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=0,
            include_intercept=True,
            d_grid=np.array([0.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert all(len(c.betas) == 1 for c in result.top_k_results)


def test_zero_deterministic_cycles_without_intercept_fits_no_betas():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=0,
            include_intercept=False,
            d_grid=np.array([0.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert all(c.betas.shape == (0,) for c in result.top_k_results)
    assert all(c.beta_standard_errors.shape == (0,) for c in result.top_k_results)


def test_top_level_detect_beta_significance_export():
    result = detect_beta_significance(
        np.array([2.0]),
        np.array([1.0]),
    )

    np.testing.assert_array_equal(result.significant, np.array([True]))


def test_all_test_values_are_finite():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert all(np.isfinite(c.test_value) and np.isfinite(c.test_star_value) for c in result.top_k_results)


def test_residuals_length_matches_T():
    T = 50
    result = run_cyclical_fractional_test(_series(T=T), config=_small_config())
    assert all(c.residuals.shape == (T,) for c in result.top_k_results)


# ---------------------------------------------------------------------------
# Config and kwargs
# ---------------------------------------------------------------------------


def test_kwargs_override_config_fields():
    config = _small_config(top_k=1)
    result = run_cyclical_fractional_test(_series(), config=config, top_k=2)
    assert len(result.top_k_results) == 2


def test_kwargs_work_without_config():
    result = run_cyclical_fractional_test(
        _series(),
        n_deterministic_cycles=2,
        d_grid=np.array([0.0]),
        r_window=0,
        top_k=1,
        stochastic_cycle_mode="single",
    )
    assert result.best_result is not None


def test_explicit_white_noise_matches_default_behavior():
    y = _series(T=60, seed=11)
    config = _small_config()
    default_result = run_cyclical_fractional_test(y, config=config)
    explicit_result = run_cyclical_fractional_test(
        y, config=config, error_model="white_noise"
    )
    assert explicit_result.r_peak == default_result.r_peak
    assert explicit_result.best_result.cycles == default_result.best_result.cycles
    assert explicit_result.best_result.ar_coefficients == ()
    assert explicit_result.best_result.xa == default_result.best_result.xa
    assert explicit_result.best_result.xaa == default_result.best_result.xaa
    assert explicit_result.best_result.test_value == default_result.best_result.test_value


@pytest.mark.parametrize("error_model, coefficient_count", [("ar1", 1), ("ar2", 2)])
def test_full_pipeline_supports_ar_error_models(error_model, coefficient_count):
    result = run_cyclical_fractional_test(
        _series(T=60, seed=123),
        config=_small_config(d_grid=np.array([0.0]), r_window=0, top_k=1),
        error_model=error_model,
    )
    assert result.best_result.error_model == error_model
    assert len(result.best_result.ar_coefficients) == coefficient_count
    assert np.isfinite(result.best_result.test_value)


def test_default_config_runs():
    result = run_cyclical_fractional_test(_series(T=60))
    assert result.best_result is not None


def test_zero_frequency_excluded_by_default_even_when_dominant():
    result = run_cyclical_fractional_test(
        _zero_dominant_series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0]),
            r_window=0,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert result.r_peak != 0
    assert 0 not in set(int(R) for R in result.r_candidates)
    assert result.best_result.cycles[0].R != 0


def test_zero_frequency_can_be_selected_when_included():
    result = run_cyclical_fractional_test(
        _zero_dominant_series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0]),
            r_window=0,
            top_k=1,
            stochastic_cycle_mode="single",
            exclude_zero_frequency=False,
        ),
    )
    assert result.r_peak == 0
    np.testing.assert_array_equal(result.r_candidates, np.array([0]))
    assert result.best_result.cycles[0].R == 0
    assert np.isfinite(result.best_result.test_value)
    assert np.isnan(result.best_result.test_star_value)


def test_multi_cycle_mode_runs_and_selects_requested_cycles():
    config = CyclicalTestConfig(
        n_deterministic_cycles=2,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.5]),
        top_k=3,
        stochastic_cycle_mode="multi_cycle",
        n_stochastic_cycles=2,
    )
    result = run_cyclical_fractional_test(_series(T=60, seed=21), config=config)
    assert result.best_result is result.top_k_results[0]
    assert len(result.top_k_results) == 3
    assert len(result.best_result.cycles) == 2
    assert [cycle.R for cycle in result.best_result.cycles] == [
        int(R) for R in result.r_candidates
    ]
    assert result.r_peak == int(result.r_candidates[0])
    assert result.n_candidates_evaluated == len(config.d_grid) ** 2
    assert result.diagnostics.r_candidates_count == 2
    assert result.diagnostics.n_stochastic_cycles == 2
    assert np.isfinite(result.best_result.test_value)


@pytest.mark.parametrize("error_model", ["white_noise", "ar1", "ar2"])
def test_multi_cycle_mode_supports_error_models(error_model):
    config = CyclicalTestConfig(
        n_deterministic_cycles=2,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        top_k=1,
        stochastic_cycle_mode="multi_cycle",
        n_stochastic_cycles=2,
        error_model=error_model,
    )
    result = run_cyclical_fractional_test(_series(T=80, seed=33), config=config)
    assert result.best_result.error_model == error_model
    assert len(result.best_result.cycles) == 2
    assert np.isfinite(result.best_result.test_value)


def test_multi_cycle_can_include_zero_frequency_when_included():
    result = run_cyclical_fractional_test(
        _zero_dominant_series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0]),
            top_k=1,
            stochastic_cycle_mode="multi_cycle",
            n_stochastic_cycles=2,
            exclude_zero_frequency=False,
        ),
    )
    cycle_Rs = [cycle.R for cycle in result.best_result.cycles]
    assert result.r_peak == 0
    assert 0 in cycle_Rs
    assert 0 in set(int(R) for R in result.r_candidates)
    assert np.isfinite(result.best_result.test_value)
    assert np.isnan(result.best_result.test_star_value)


def test_zero_frequency_degenerate_variance_rejects_test_star_ranking():
    with pytest.raises(InvalidConfigurationError):
        run_cyclical_fractional_test(
            _zero_dominant_series(T=60),
            config=CyclicalTestConfig(
                n_deterministic_cycles=2,
                d_search_strategy="fixed_grid",
                d_grid=np.array([0.0]),
                r_window=0,
                top_k=1,
                stochastic_cycle_mode="single",
                exclude_zero_frequency=False,
                statistic_mode="test_star",
            ),
        )


def test_ignored_stochastic_rs_excludes_single_cycle_peak_and_grid_candidate():
    result = run_cyclical_fractional_test(
        _cyclic_series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=0,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0]),
            r_window=0,
            stochastic_cycle_mode="single",
            ignored_stochastic_rs=(4,),
        ),
    )

    assert result.r_peak == 7
    assert list(result.r_candidates) == [7]
    assert result.best_result.cycles[0].R == 7


def test_ignored_stochastic_rs_excludes_multi_cycle_top_peak():
    result = run_cyclical_fractional_test(
        _cyclic_series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=0,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0]),
            top_k=1,
            stochastic_cycle_mode="multi_cycle",
            n_stochastic_cycles=2,
            ignored_stochastic_rs=(4,),
        ),
    )

    assert tuple(result.r_candidates) == (7, 11)
    assert [cycle.R for cycle in result.best_result.cycles] == [7, 11]


def test_ignored_stochastic_rs_rejects_out_of_range_frequency():
    with pytest.raises(InvalidConfigurationError):
        run_cyclical_fractional_test(
            _series(T=30),
            config=CyclicalTestConfig(ignored_stochastic_rs=(30,)),
        )


def test_ignored_stochastic_rs_rejects_when_no_search_frequencies_remain():
    with pytest.raises(InvalidConfigurationError):
        run_cyclical_fractional_test(
            _series(T=10),
            config=CyclicalTestConfig(
                exclude_zero_frequency=False,
                ignored_stochastic_rs=tuple(range(5)),
            ),
        )


def test_multi_cycle_fixed_grid_evaluates_joint_d_candidates(monkeypatch):
    target_d = (0.0, 0.5)

    def fake_evaluate_candidate(y, X, cycles, config):
        assert len(cycles) == 2
        d_values = tuple(cycle.D for cycle in cycles)
        score = 0.0 if d_values == target_d else 10.0
        return GridCandidateResult(
            cycles=tuple(cycles),
            error_model=config.error_model,
            test_value=score,
            test_star_value=score,
            xa=score,
            xaa=1.0,
            variance_time=1.0,
            variance_frequency=1.0,
        )

    monkeypatch.setattr(
        "cyclical_fractional_test.api.evaluate_candidate",
        fake_evaluate_candidate,
    )
    result = run_cyclical_fractional_test(
        _series(T=60, seed=41),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0, 0.5]),
            top_k=1,
            stochastic_cycle_mode="multi_cycle",
            n_stochastic_cycles=2,
        ),
    )
    assert tuple(cycle.D for cycle in result.best_result.cycles) == target_d
    assert result.n_candidates_evaluated == 4


def test_multi_cycle_adaptive_evaluates_joint_d_candidates(monkeypatch):
    target_d = (0.25, 0.5)

    def fake_evaluate_candidate(y, X, cycles, config):
        assert len(cycles) == 2
        d_values = tuple(cycle.D for cycle in cycles)
        if d_values == target_d:
            score = 0.0
        elif d_values == (0.0, 0.5):
            score = 1.0
        else:
            score = 10.0
        return GridCandidateResult(
            cycles=tuple(cycles),
            error_model=config.error_model,
            test_value=score,
            test_star_value=score,
            xa=score,
            xaa=1.0,
            variance_time=1.0,
            variance_frequency=1.0,
        )

    monkeypatch.setattr(
        "cyclical_fractional_test.api.evaluate_candidate",
        fake_evaluate_candidate,
    )
    result = run_cyclical_fractional_test(
        _series(T=60, seed=41),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="adaptive",
            d_coarse_grid=np.array([0.0, 0.5, 1.0]),
            d_fine_step=0.25,
            d_fine_radius=0.25,
            top_k=1,
            stochastic_cycle_mode="multi_cycle",
            n_stochastic_cycles=2,
        ),
    )
    assert tuple(cycle.D for cycle in result.best_result.cycles) == target_d
    assert result.diagnostics.n_coarse_evaluations == 9
    assert result.diagnostics.n_fine_evaluations == 5
    assert result.n_candidates_evaluated == 14


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_series", [
    np.array([1.0, float("nan"), 3.0, 4.0, 5.0]),
    np.array([1.0, float("inf"), 3.0, 4.0, 5.0]),
    np.ones((10, 2)),
    np.array([1.0, 2.0]),
])
def test_rejects_invalid_series(bad_series):
    with pytest.raises((InvalidSeriesError, ValueError)):
        run_cyclical_fractional_test(bad_series, config=_small_config())


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_rejects_invalid_top_k(bad_top_k):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        run_cyclical_fractional_test(_series(T=30), config=_small_config(top_k=bad_top_k))


def test_rejects_non_finite_d_grid():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        run_cyclical_fractional_test(
            _series(T=30),
            config=_small_config(d_grid=np.array([0.0, float("nan")])),
        )


def test_rejects_invalid_error_model():
    with pytest.raises(InvalidConfigurationError):
        run_cyclical_fractional_test(
            _series(T=30),
            config=_small_config(error_model="ar3"),
        )


# ---------------------------------------------------------------------------
# Top-k ordering and counting
# ---------------------------------------------------------------------------


def test_top_k_ordered_ascending_by_abs_test_value():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=42),
        config=_small_config(d_grid=np.array([0.0, 0.3, 0.7, 1.0]), r_window=2, top_k=4),
    )
    scores = [abs(c.test_value) for c in result.top_k_results]
    assert scores == sorted(scores)


def test_n_candidates_evaluated_matches_grid_size():
    result = run_cyclical_fractional_test(
        _series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0, 1.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert result.n_candidates_evaluated == len(result.r_candidates) * 2


def test_fewer_candidates_than_top_k_returns_all():
    result = run_cyclical_fractional_test(
        _series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0]),
            r_window=0,
            top_k=10,
            stochastic_cycle_mode="single",
        ),
    )
    assert len(result.top_k_results) == 1
    assert result.n_candidates_evaluated == 1


# ---------------------------------------------------------------------------
# Adaptive D search
# ---------------------------------------------------------------------------


def test_default_run_uses_adaptive_search():
    result = run_cyclical_fractional_test(_series(T=60), config=_small_config(r_window=1))
    assert result.config.d_search_strategy == "adaptive"
    assert result.diagnostics.d_search_strategy == "adaptive"
    assert result.best_result is not None


def test_explicit_adaptive_strategy_runs():
    result = run_cyclical_fractional_test(
        _series(T=60), config=_small_config(r_window=1), d_search_strategy="adaptive"
    )
    assert result.best_result is not None
    assert result.diagnostics.n_coarse_evaluations is not None
    assert result.diagnostics.n_fine_evaluations is not None


def test_fixed_grid_strategy_preserves_old_behavior():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0, 0.5, 1.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert result.n_candidates_evaluated == len(result.r_candidates) * 3
    assert result.diagnostics.d_search_strategy == "fixed_grid"


def test_adaptive_n_candidates_counts_coarse_plus_fine_minus_reuse():
    result = run_cyclical_fractional_test(
        _series(T=60), config=_small_config(r_window=0, top_k=1)
    )
    diag = result.diagnostics
    # Single R: 11 coarse values. The fine grid overlaps the best coarse value, so
    # the reused candidate is not recounted. Fine count is 18 for an interior best
    # coarse D, or 9 when it lands on a [0,1] boundary (one-sided window).
    assert diag.n_coarse_evaluations == 11
    assert 9 <= diag.n_fine_evaluations <= 18
    assert result.n_candidates_evaluated == diag.n_coarse_evaluations + diag.n_fine_evaluations


def test_adaptive_best_d_lies_on_coarse_or_fine_grid():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=7), config=_small_config(r_window=0, top_k=1)
    )
    best_d = result.best_result.cycles[0].D
    coarse = set(np.round(np.linspace(0.0, 1.0, 11), 12))
    best_coarse = result.diagnostics.best_coarse_d_per_r[0]
    fine = set(np.round(np.arange(round(best_coarse - 0.09, 12), best_coarse + 0.09 + 0.005, 0.01), 12))
    fine = {v for v in fine if 0.0 <= v <= 1.0}
    assert round(best_d, 12) in (coarse | fine)


def test_adaptive_refines_around_best_coarse():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=7), config=_small_config(r_window=0, top_k=1)
    )
    best_d = result.best_result.cycles[0].D
    best_coarse = result.diagnostics.best_coarse_d_per_r[0]
    # Final D must be within the local fine window around the best coarse D.
    assert abs(best_d - best_coarse) <= 0.09 + 1e-9


@pytest.mark.parametrize("error_model", ["white_noise", "ar1", "ar2"])
def test_adaptive_supports_error_models(error_model):
    result = run_cyclical_fractional_test(
        _series(T=60, seed=3),
        config=_small_config(r_window=0, top_k=1),
        error_model=error_model,
    )
    assert result.best_result.error_model == error_model
    assert np.isfinite(result.best_result.test_value)


def test_multi_cycle_adaptive_joint_search_runs():
    result = run_cyclical_fractional_test(
        _series(T=70, seed=13),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="adaptive",
            d_coarse_grid=np.array([0.0, 0.5, 1.0]),
            d_fine_step=0.25,
            d_fine_radius=0.25,
            top_k=2,
            stochastic_cycle_mode="multi_cycle",
            n_stochastic_cycles=2,
        ),
    )
    diag = result.diagnostics
    assert len(result.best_result.cycles) == 2
    assert all(len(candidate.cycles) == 2 for candidate in result.top_k_results)
    assert diag.n_coarse_evaluations == 3 ** 2
    assert diag.n_fine_evaluations is not None
    assert result.n_candidates_evaluated == (
        diag.n_coarse_evaluations + diag.n_fine_evaluations
    )
    assert [cycle.D for cycle in result.best_result.cycles] == diag.final_d_per_r
    assert len(diag.best_coarse_d_per_r) == 2


def test_multi_cycle_fixed_grid_counts_joint_d_product():
    result = run_cyclical_fractional_test(
        _series(T=70, seed=13),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_search_strategy="fixed_grid",
            d_grid=np.array([0.0, 0.5, 1.0]),
            top_k=1,
            stochastic_cycle_mode="multi_cycle",
            n_stochastic_cycles=2,
        ),
    )
    diag = result.diagnostics
    assert len(result.best_result.cycles) == 2
    assert diag.n_coarse_evaluations is None
    assert diag.n_fine_evaluations is None
    assert result.n_candidates_evaluated == 3 ** 2


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_present_and_correct_type():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert isinstance(result.diagnostics, TestDiagnostics)


def test_diagnostics_counters_consistent():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    diag = result.diagnostics
    assert diag.n_candidates_evaluated > 0
    assert diag.n_valid_candidates == diag.n_candidates_evaluated
    assert diag.n_failed_candidates == 0


def test_diagnostics_r_peak_and_candidates_match_result():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert result.diagnostics.r_peak == result.r_peak
    assert result.diagnostics.r_candidates_count == len(result.r_candidates)


def test_diagnostics_periodogram_summary_consistent():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    ps = result.diagnostics.periodogram_summary
    assert isinstance(ps, PeriodogramSummary)
    assert ps.peak_index == result.r_peak


def test_diagnostics_does_not_affect_ordering():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=42),
        config=_small_config(d_grid=np.array([0.0, 0.5]), r_window=2, top_k=3),
    )
    assert result.best_result is result.top_k_results[0]
    scores = [abs(c.test_value) for c in result.top_k_results]
    assert scores == sorted(scores)


def test_diagnostics_default_d_grid_has_11_values():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_grid=None,
            r_window=0,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert result.diagnostics.d_grid_count == 11


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_full_pipeline_with_structured_series():
    from cyclical_fractional_test.chebyshev import build_chebyshev_design

    rng = np.random.default_rng(2024)
    T = 30
    X_ref = build_chebyshev_design(T, 2)
    y = 0.5 * X_ref[:, 0] + 0.2 * X_ref[:, 1] + 0.1 * rng.standard_normal(T)

    config = CyclicalTestConfig(
        n_deterministic_cycles=2,
        include_intercept=False,
        d_grid=np.array([0.0, 0.5]),
        r_window=1,
        top_k=2,
        stochastic_cycle_mode="single",
    )
    result = run_cyclical_fractional_test(y, config=config)

    assert isinstance(result, CyclicalFractionalTestResult)
    assert result.best_result is result.top_k_results[0]
    for cand in result.top_k_results:
        assert np.isfinite(cand.test_value)
        assert cand.betas.shape == (2,)
        assert cand.residuals.shape == (T,)
    scores = [abs(c.test_value) for c in result.top_k_results]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Under-threshold results
# ---------------------------------------------------------------------------


def test_threshold_none_yields_no_under_threshold_results():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert result.under_threshold_results is None


def test_threshold_groups_all_candidates_by_frequency_tuple_when_large():
    config = _fixed_config(d_grid=np.array([0.0, 0.3, 0.7, 1.0]))
    result = run_cyclical_fractional_test(_series(T=60, seed=1), config=config, threshold=1e9)
    utr = result.under_threshold_results
    assert utr is not None
    # A huge threshold keeps every evaluated candidate; single-cycle keys are (R,).
    assert set(utr.keys()) == {(int(r),) for r in result.r_candidates}
    assert sum(len(cands) for cands in utr.values()) == result.n_candidates_evaluated
    assert all(abs(c.test_value) < 1e9 for cands in utr.values() for c in cands)


def test_threshold_keys_and_scores_are_sorted():
    config = _fixed_config(d_grid=np.array([0.0, 0.3, 0.7, 1.0]))
    result = run_cyclical_fractional_test(_series(T=60, seed=9), config=config, threshold=1e9)
    utr = result.under_threshold_results
    assert list(utr.keys()) == sorted(utr.keys())
    for cands in utr.values():
        scores = [abs(c.test_value) for c in cands]
        assert scores == sorted(scores)


def test_threshold_filters_strictly_below():
    config = _fixed_config(d_grid=np.array([0.0, 0.3, 0.7, 1.0]))
    y = _series(T=60, seed=5)
    full = run_cyclical_fractional_test(y, config=config, threshold=1e9).under_threshold_results
    thr = 0.7
    small = run_cyclical_fractional_test(y, config=config, threshold=thr).under_threshold_results
    # Every retained candidate is strictly below the threshold.
    assert all(abs(c.test_value) < thr for cands in small.values() for c in cands)
    # Nothing qualifying is dropped: count matches a manual filter of the full set.
    expected = sum(1 for cands in full.values() for c in cands if abs(c.test_value) < thr)
    assert sum(len(cands) for cands in small.values()) == expected


def test_threshold_adaptive_collects_multiple_d_per_r():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=7),
        config=_small_config(r_window=0, top_k=1),
        threshold=1e9,
    )
    utr = result.under_threshold_results
    assert utr is not None
    # Single R (r_window=0); a huge threshold keeps every evaluated D for key (R,).
    assert len(utr) == 1
    (key,) = utr.keys()
    assert len(key) == 1
    (cands,) = utr.values()
    assert len(cands) == result.n_candidates_evaluated
    assert len(cands) > 1
    assert len({c.cycles[0].R for c in cands}) == 1
    assert len({round(c.cycles[0].D, 12) for c in cands}) == len(cands)


def test_threshold_can_store_lightweight_candidates_without_residuals():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=7),
        config=_small_config(
            r_window=0,
            top_k=1,
            return_residuals_for_threshold=False,
        ),
        threshold=1e9,
    )
    grouped = result.under_threshold_results
    assert grouped is not None
    candidates = [candidate for bucket in grouped.values() for candidate in bucket]
    assert len(candidates) == result.n_candidates_evaluated
    assert all(candidate.betas is not None for candidate in candidates)
    assert all(candidate.beta_standard_errors is not None for candidate in candidates)
    assert all(candidate.residuals is None for candidate in candidates)
    assert result.best_result is not None
    assert result.best_result.residuals is not None


def test_threshold_multi_cycle_groups_joint_candidates_by_full_r_tuple():
    config = CyclicalTestConfig(
        n_deterministic_cycles=2,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.5]),
        top_k=3,
        stochastic_cycle_mode="multi_cycle",
        n_stochastic_cycles=2,
    )

    result = run_cyclical_fractional_test(
        _series(T=60, seed=11), config=config, threshold=1e9
    )

    utr = result.under_threshold_results
    assert utr is not None
    expected_key = tuple(int(r) for r in result.r_candidates)
    assert set(utr.keys()) == {expected_key}
    assert sum(len(cands) for cands in utr.values()) == result.n_candidates_evaluated
    (cands,) = utr.values()
    assert all(len(candidate.cycles) == 2 for candidate in cands)
    assert all(
        tuple(cycle.R for cycle in candidate.cycles) == expected_key
        for candidate in cands
    )
    scores = [abs(candidate.test_value) for candidate in cands]
    assert scores == sorted(scores)


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0, float("nan"), float("inf"), True])
def test_rejects_invalid_threshold(bad_threshold):
    with pytest.raises(InvalidConfigurationError):
        run_cyclical_fractional_test(_series(T=30), config=_small_config(), threshold=bad_threshold)
