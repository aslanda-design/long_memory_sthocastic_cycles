import copy

import numpy as np
import pytest

from cyclical_fractional_test import (
    CyclicalFractionalModel,
    CyclicalTestConfig,
    NotFittedError,
)
from cyclical_fractional_test.exceptions import InvalidConfigurationError


def _series(n=120, freq=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    return np.cos(2 * np.pi * freq * t / n) + 0.3 * rng.standard_normal(n)


def test_fit_returns_self_and_sets_attributes():
    y = _series()
    model = CyclicalFractionalModel()
    out = model.fit(y)
    assert out is model
    assert model.n_train_ == len(y)
    assert isinstance(model.R_, int)
    assert 0.0 <= model.D_ <= 1.0
    assert model.betas_.ndim == 1
    assert model.beta_standard_errors_.shape == model.betas_.shape
    assert model.beta_t_statistics_.shape == model.betas_.shape
    assert model.beta_significant_.shape == model.betas_.shape
    assert model.residuals_.shape == (len(y),)
    assert model.innovation_variance_ > 0.0


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        CyclicalFractionalModel().predict(5)


def test_predict_interval_before_fit_raises():
    with pytest.raises(NotFittedError):
        CyclicalFractionalModel().predict_interval(5)


def test_detect_beta_significance_before_fit_raises():
    with pytest.raises(NotFittedError):
        CyclicalFractionalModel().detect_beta_significance()


def test_predict_in_sample_length():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    assert model.predict(len(y)).shape == (len(y),)
    assert model.predict(10).shape == (10,)


def test_predict_out_of_sample_length():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    out = model.predict(len(y) + 15)
    assert out.shape == (len(y) + 15,)
    assert np.all(np.isfinite(out))


def test_predict_prefix_consistency():
    # predict(n) for n <= T must be the prefix of the full in-sample reconstruction.
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    full = model.predict(len(y))
    np.testing.assert_allclose(model.predict(30), full[:30])


def test_predict_extends_in_sample_prefix():
    # The first T entries of an out-of-sample prediction equal the in-sample ones.
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    in_sample = model.predict(len(y))
    extended = model.predict(len(y) + 8)
    np.testing.assert_allclose(extended[: len(y)], in_sample)


def test_predict_recursively_does_not_use_observed_values_after_start():
    y = _series()
    model = CyclicalFractionalModel(error_model="ar1").fit(y)
    start = 70

    recursive = model.predict_recursively(len(y), start=start)
    altered = copy.deepcopy(model)
    altered.y_train_[start:] = altered.y_train_[start:] + 100.0
    altered.residuals_[start:] = altered.residuals_[start:] - 100.0

    np.testing.assert_allclose(
        altered.predict_recursively(len(y), start=start),
        recursive,
        atol=1e-10,
    )
    np.testing.assert_allclose(recursive[:start], model.predict(len(y))[:start])
    assert recursive.shape == (len(y),)
    assert np.all(np.isfinite(recursive))


@pytest.mark.parametrize("error_model", ["white_noise", "ar1", "ar2"])
def test_predict_all_error_models(error_model):
    y = _series(seed=2)
    model = CyclicalFractionalModel(error_model=error_model).fit(y)
    expected_order = {"white_noise": 0, "ar1": 1, "ar2": 2}[error_model]
    assert len(model.ar_coefficients_) == expected_order
    out = model.predict(len(y) + 5)
    assert np.all(np.isfinite(out))


def test_kwargs_override_config():
    model = CyclicalFractionalModel(error_model="ar1", n_deterministic_cycles=3)
    assert model.config.error_model == "ar1"
    assert model.config.n_deterministic_cycles == 3


def test_fit_predict_supports_explicit_chebyshev_orders():
    y = _series(seed=13)
    model = CyclicalFractionalModel(
        n_deterministic_cycles=99,
        chebyshev_orders=(2, 5),
        include_intercept=True,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        r_window=0,
    ).fit(y)

    assert model.X_train_.shape == (len(y), 3)
    assert model.betas_.shape == (3,)
    out = model.predict(len(y) + 5)
    assert out.shape == (len(y) + 5,)
    assert np.all(np.isfinite(out))


def test_detect_beta_significance_returns_model_t_tests():
    y = _series(seed=14)
    model = CyclicalFractionalModel(
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        r_window=0,
    ).fit(y)

    result = model.detect_beta_significance()

    np.testing.assert_allclose(result.betas, model.betas_)
    np.testing.assert_allclose(result.standard_errors, model.beta_standard_errors_)
    np.testing.assert_allclose(result.t_statistics, model.beta_t_statistics_)
    np.testing.assert_array_equal(result.significant, model.beta_significant_)
    assert result.critical_value == pytest.approx(1.645)


def test_detect_beta_significance_recomputes_missing_candidate_standard_errors():
    y = _series(seed=15)
    model = CyclicalFractionalModel(
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        r_window=0,
    ).fit(y)
    candidate = copy.deepcopy(model.result_.best_result)
    candidate.beta_standard_errors = None

    result = model.detect_beta_significance(candidate=candidate)

    np.testing.assert_allclose(result.betas, model.betas_)
    np.testing.assert_allclose(result.standard_errors, model.beta_standard_errors_)


def test_detect_beta_significance_uses_lightweight_candidate_standard_errors():
    y = _series(seed=16)
    model = CyclicalFractionalModel(
        threshold=1e9,
        return_residuals_for_threshold=False,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        r_window=0,
    ).fit(y)
    candidate = model.get_under_threshold_candidates()[0]

    assert candidate.residuals is None
    result = model.detect_beta_significance(candidate=candidate)

    assert result.betas.shape == model.betas_.shape
    assert result.standard_errors.shape == model.beta_standard_errors_.shape


@pytest.mark.parametrize("include_intercept, expected_betas", [(True, 1), (False, 0)])
def test_fit_predict_supports_zero_deterministic_cycles(
    include_intercept, expected_betas
):
    y = _series(seed=8)
    model = CyclicalFractionalModel(
        n_deterministic_cycles=0,
        include_intercept=include_intercept,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        r_window=0,
    ).fit(y)
    assert model.betas_.shape == (expected_betas,)
    out = model.predict(len(y) + 5)
    assert out.shape == (len(y) + 5,)
    assert np.all(np.isfinite(out))


def test_explicit_config_respected():
    config = CyclicalTestConfig(error_model="ar2")
    model = CyclicalFractionalModel(config=config)
    assert model.config.error_model == "ar2"


def test_predict_interval_brackets_prediction_and_widens():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    n = len(y) + 12
    center = model.predict(n)
    lower, upper = model.predict_interval(n, alpha=0.05)
    assert np.all(lower < center) and np.all(center < upper)
    # Forecast bounds should not shrink as the horizon grows.
    widths = upper[len(y):] - lower[len(y):]
    assert np.all(np.diff(widths) >= -1e-9)


def test_predict_interval_rejects_unknown_alpha():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    with pytest.raises(InvalidConfigurationError):
        model.predict_interval(5, alpha=0.5)


def test_predict_rejects_non_positive_n():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    with pytest.raises(InvalidConfigurationError):
        model.predict(0)


def test_predict_recursively_rejects_invalid_start():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    with pytest.raises(InvalidConfigurationError):
        model.predict_recursively(len(y), start=0)
    with pytest.raises(InvalidConfigurationError):
        model.predict_recursively(len(y), start=len(y))


def test_multi_cycle_predict():
    y = _series(seed=4)
    model = CyclicalFractionalModel(
        stochastic_cycle_mode="multi_cycle", n_stochastic_cycles=2
    ).fit(y)
    out = model.predict(len(y) + 6)
    assert out.shape == (len(y) + 6,)
    assert np.all(np.isfinite(out))


def test_under_threshold_candidates_empty_without_threshold():
    y = _series(seed=9)
    model = CyclicalFractionalModel().fit(y)

    assert model.get_under_threshold_candidates() == []


def test_under_threshold_candidates_are_prediction_ready():
    y = _series(seed=10)
    model = CyclicalFractionalModel(
        threshold=1e9,
        return_residuals_for_threshold=True,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.5]),
        r_window=0,
    ).fit(y)

    candidates = model.get_under_threshold_candidates()

    assert len(candidates) == model.result_.n_candidates_evaluated
    assert candidates
    candidate = candidates[-1]
    assert candidate.betas is not None
    assert candidate.residuals is not None
    assert candidate.residuals.shape == (len(y),)
    assert isinstance(candidate.ar_coefficients, tuple)

    prediction = model.predict_with_candidate(candidate, len(y) + 5)
    lower, upper = model.predict_interval_with_candidate(candidate, len(y) + 5)

    assert prediction.shape == (len(y) + 5,)
    assert lower.shape == prediction.shape
    assert upper.shape == prediction.shape
    assert np.all(np.isfinite(prediction))
    assert np.all(lower < prediction)
    assert np.all(prediction < upper)


def test_predict_with_best_candidate_matches_predict():
    y = _series(seed=12)
    model = CyclicalFractionalModel(
        threshold=1e9,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.5]),
        r_window=0,
    ).fit(y)

    best = model.result_.best_result

    assert best is not None
    np.testing.assert_allclose(
        model.predict_with_candidate(best, len(y) + 7),
        model.predict(len(y) + 7),
    )
