from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

import numpy as np

from .api import run_cyclical_fractional_test
from .chebyshev import build_chebyshev_design, build_chebyshev_design_at
from .config import CyclicalTestConfig
from .exceptions import InvalidConfigurationError, NotFittedError
from .filters import filter_response_and_design
from .prediction import (
    compute_ma_weights,
    forecast_out_of_sample,
    reconstruct_in_sample,
)
from .regression import (
    BetaSignificanceResult,
    DEFAULT_BETA_SIGNIFICANCE_CRITICAL_VALUE,
    compute_beta_standard_errors,
    detect_beta_significance as detect_beta_significance_from_arrays,
    estimate_innovation_variance,
)
from .results import CyclicalFractionalTestResult, GridCandidateResult, StochasticCycle
from .scoring import score_candidate
from .validation import validate_series

_Z_FOR_TWO_SIDED = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}


class CyclicalFractionalModel:
    """Scikit-learn-style estimator for fractional cyclic long memory.

    `fit` runs the cyclical fractional test on a series and stores the selected
    model (best cycle tuple, OLS coefficients, AR error coefficients, residuals).
    `predict(n)` returns the reconstructed series for t = 1, ..., n: in-sample
    one-step-ahead values within the observed range, and out-of-sample forecasts
    beyond the training length. Hyperparameters mirror CyclicalTestConfig and can
    be passed directly or via **kwargs (overriding the config).
    """

    def __init__(
        self,
        config: Optional[CyclicalTestConfig] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = CyclicalTestConfig()
        if kwargs:
            config = dataclasses.replace(config, **kwargs)
        self.config = config
        self.threshold = threshold

    def fit(self, y: Any) -> "CyclicalFractionalModel":
        """Run the test on y and store the selected model. Returns self."""
        arr = validate_series(y)
        result = run_cyclical_fractional_test(
            arr, config=self.config, threshold=self.threshold
        )
        best = result.best_result
        if best is None or best.betas is None or best.residuals is None:
            raise InvalidConfigurationError(
                "fit did not produce a usable candidate; check the configuration."
            )

        T = len(arr)
        self.result_ = result
        self.y_train_ = arr
        self.n_train_ = T
        self.X_train_ = build_chebyshev_design(
            T,
            self.config.n_deterministic_cycles,
            self.config.include_intercept,
            self.config.chebyshev_orders,
        )
        self.cycles_ = best.cycles
        self.R_ = best.cycles[0].R
        self.D_ = best.cycles[0].D
        self.betas_ = np.asarray(best.betas, dtype=float)
        self.residuals_ = np.asarray(best.residuals, dtype=float)
        self.error_model_ = best.error_model
        self.ar_coefficients_ = np.asarray(best.ar_coefficients, dtype=float)
        beta_standard_errors = getattr(best, "beta_standard_errors", None)
        if beta_standard_errors is None:
            beta_standard_errors = self._compute_beta_standard_errors_from_candidate(best)
        self.beta_standard_errors_ = np.asarray(beta_standard_errors, dtype=float)
        self.beta_significance_ = detect_beta_significance_from_arrays(
            self.betas_, self.beta_standard_errors_
        )
        self.beta_t_statistics_ = self.beta_significance_.t_statistics
        self.beta_significant_ = self.beta_significance_.significant
        self.innovation_variance_ = estimate_innovation_variance(
            self.residuals_, self.ar_coefficients_
        )
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return the model's reconstruction of the series for t = 1, ..., n.

        For n <= T these are in-sample one-step-ahead values; for n > T the tail
        T+1, ..., n is forecast out of sample.
        """
        self._check_fitted()
        _validate_n(n)
        return self._predict_from_candidate(self._best_candidate(), int(n))

    def get_under_threshold_candidates(self) -> list[GridCandidateResult]:
        """Return all under-threshold candidates sorted globally by score."""
        self._check_fitted()
        grouped = self.result_.under_threshold_results
        if not grouped:
            return []
        candidates = [candidate for bucket in grouped.values() for candidate in bucket]
        return sorted(
            candidates,
            key=lambda candidate: score_candidate(candidate, self.config.statistic_mode),
        )

    def detect_beta_significance(
        self,
        critical_value: float = DEFAULT_BETA_SIGNIFICANCE_CRITICAL_VALUE,
        candidate: GridCandidateResult | None = None,
    ) -> BetaSignificanceResult:
        """Return t-test decisions for deterministic beta coefficients.

        By default this uses the fitted best model. When a candidate is provided,
        stored standard errors are used if present; otherwise they are recomputed
        from the fitted training design and that candidate's filtered residuals.
        """
        self._check_fitted()
        if candidate is None:
            standard_errors = getattr(self, "beta_standard_errors_", None)
            if standard_errors is None:
                standard_errors = self._compute_beta_standard_errors_from_candidate(
                    self._best_candidate()
                )
                self.beta_standard_errors_ = standard_errors
            return detect_beta_significance_from_arrays(
                self.betas_, standard_errors, critical_value
            )

        if not isinstance(candidate, GridCandidateResult):
            raise InvalidConfigurationError(
                "candidate must be a GridCandidateResult produced by the test."
            )
        if candidate.betas is None:
            raise InvalidConfigurationError("candidate must include betas.")
        betas = np.asarray(candidate.betas, dtype=float)
        standard_errors = getattr(candidate, "beta_standard_errors", None)
        if standard_errors is None:
            standard_errors = self._compute_beta_standard_errors_from_candidate(
                candidate
            )
        return detect_beta_significance_from_arrays(
            betas, standard_errors, critical_value
        )

    def predict_with_candidate(
        self, candidate: GridCandidateResult, n: int
    ) -> np.ndarray:
        """Predict with a stored candidate without changing the fitted best model."""
        self._check_fitted()
        _validate_n(n)
        return self._predict_from_candidate(candidate, int(n))

    def predict_recursively(self, n: int, start: int) -> np.ndarray:
        """Return predictions that become recursive at a zero-based cutoff.

        Values before `start` are the usual in-sample one-step-ahead
        reconstruction. From `start` onward, observed training values are no
        longer fed into the recursion; the path is generated from the model state
        available before `start` and then from its own forecasts.
        """
        self._check_fitted()
        _validate_n(n)
        n = int(n)
        start = _validate_recursive_start(
            start,
            n=n,
            n_train=self.n_train_,
            ar_order=len(self.ar_coefficients_),
        )

        mode = self.config.stochastic_cycle_mode
        in_sample = reconstruct_in_sample(
            self.y_train_,
            self.X_train_,
            self.cycles_,
            self.betas_,
            self.residuals_,
            self.ar_coefficients_,
            mode,
        )
        horizon = n - start
        t_forecast = np.arange(start + 1, n + 1, dtype=float)
        X_forecast = build_chebyshev_design_at(
            t_forecast,
            self.n_train_,
            self.config.n_deterministic_cycles,
            self.config.include_intercept,
            self.config.chebyshev_orders,
        )
        forecast = forecast_out_of_sample(
            self.y_train_[:start],
            self.X_train_[:start],
            X_forecast,
            self.cycles_,
            self.betas_,
            self.residuals_[:start],
            self.ar_coefficients_,
            mode,
            horizon,
            coefficient_t_ref=self.n_train_,
        )
        return np.concatenate([in_sample[:start], forecast])

    def predict_interval(
        self, n: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction bounds for predict(n) at level 1 − alpha.

        In-sample bounds use the one-step innovation standard deviation σ̂. Forecast
        bounds widen with the accumulated MA(∞) weights of the inverse cyclic filter
        and AR error: the horizon-k variance is σ̂² Σ_{l<k} ψ_l². Parameter-estimation
        uncertainty is not included.
        """
        self._check_fitted()
        _validate_n(n)
        return self._predict_interval_from_candidate(
            self._best_candidate(), int(n), alpha
        )

    def predict_interval_with_candidate(
        self, candidate: GridCandidateResult, n: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return prediction bounds for a stored candidate."""
        self._check_fitted()
        _validate_n(n)
        return self._predict_interval_from_candidate(candidate, int(n), alpha)

    def _future_design(self, horizon: int) -> np.ndarray:
        t_future = np.arange(self.n_train_ + 1, self.n_train_ + horizon + 1, dtype=float)
        return build_chebyshev_design_at(
            t_future,
            self.n_train_,
            self.config.n_deterministic_cycles,
            self.config.include_intercept,
            self.config.chebyshev_orders,
        )

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise NotFittedError(
                "This CyclicalFractionalModel is not fitted yet. Call fit before predict."
            )

    def _best_candidate(self) -> GridCandidateResult:
        best = self.result_.best_result
        if best is None:
            raise InvalidConfigurationError(
                "Fitted result does not contain a best candidate."
            )
        return best

    def _predict_from_candidate(
        self, candidate: GridCandidateResult, n: int
    ) -> np.ndarray:
        cycles, betas, residuals, ar_coefficients = self._candidate_prediction_state(
            candidate
        )
        mode = self.config.stochastic_cycle_mode

        in_sample = reconstruct_in_sample(
            self.y_train_,
            self.X_train_,
            cycles,
            betas,
            residuals,
            ar_coefficients,
            mode,
        )
        if n <= self.n_train_:
            return in_sample[:n]

        horizon = n - self.n_train_
        X_future = self._future_design(horizon)
        forecast = forecast_out_of_sample(
            self.y_train_,
            self.X_train_,
            X_future,
            cycles,
            betas,
            residuals,
            ar_coefficients,
            mode,
            horizon,
        )
        return np.concatenate([in_sample, forecast])

    def _predict_interval_from_candidate(
        self, candidate: GridCandidateResult, n: int, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        cycles, _, residuals, ar_coefficients = self._candidate_prediction_state(
            candidate
        )
        z = _z_score(alpha)
        center = self._predict_from_candidate(candidate, n)
        innovation_variance = estimate_innovation_variance(residuals, ar_coefficients)
        sigma = float(np.sqrt(innovation_variance))

        std = np.full(n, sigma, dtype=float)
        if n > self.n_train_:
            horizon = n - self.n_train_
            weights = compute_ma_weights(
                cycles,
                ar_coefficients,
                self.config.stochastic_cycle_mode,
                self.n_train_,
                horizon,
            )
            cumulative = np.sqrt(np.cumsum(weights ** 2))
            std[self.n_train_ :] = sigma * cumulative
        return center - z * std, center + z * std

    def _candidate_prediction_state(
        self, candidate: GridCandidateResult
    ) -> Tuple[Tuple[StochasticCycle, ...], np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(candidate, GridCandidateResult):
            raise InvalidConfigurationError(
                "candidate must be a GridCandidateResult produced by the test."
            )
        if candidate.betas is None or candidate.residuals is None:
            raise InvalidConfigurationError(
                "candidate must include betas and residuals to predict."
            )
        cycles = tuple(candidate.cycles)
        if not cycles:
            raise InvalidConfigurationError("candidate.cycles must not be empty.")
        betas = np.asarray(candidate.betas, dtype=float)
        residuals = np.asarray(candidate.residuals, dtype=float)
        ar_coefficients = np.asarray(candidate.ar_coefficients, dtype=float)
        return cycles, betas, residuals, ar_coefficients

    def _compute_beta_standard_errors_from_candidate(
        self, candidate: GridCandidateResult
    ) -> np.ndarray:
        cycles, betas, residuals, _ = self._candidate_prediction_state(candidate)
        _, X_filtered = filter_response_and_design(
            self.y_train_,
            self.X_train_,
            cycles,
            mode=self.config.stochastic_cycle_mode,
        )
        if X_filtered.shape[1] != len(betas):
            raise InvalidConfigurationError(
                "candidate beta count does not match the fitted deterministic design."
            )
        return compute_beta_standard_errors(X_filtered, residuals)


def _validate_n(n: Any) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise InvalidConfigurationError(f"n must be an int, got {type(n).__name__}.")
    if int(n) < 1:
        raise InvalidConfigurationError(f"n must be >= 1, got {n}.")


def _validate_recursive_start(start: Any, n: int, n_train: int, ar_order: int) -> int:
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise InvalidConfigurationError(
            f"start must be an int, got {type(start).__name__}."
        )
    start = int(start)
    if start < 1:
        raise InvalidConfigurationError(f"start must be >= 1, got {start}.")
    if start >= n:
        raise InvalidConfigurationError(
            f"start must be smaller than n={n}, got {start}."
        )
    if start > n_train:
        raise InvalidConfigurationError(
            f"start must be <= fitted training length {n_train}, got {start}."
        )
    if start <= ar_order:
        raise InvalidConfigurationError(
            f"start must be greater than AR order {ar_order}, got {start}."
        )
    return start


def _z_score(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)):
        raise InvalidConfigurationError(
            f"alpha must be a float, got {type(alpha).__name__}."
        )
    alpha = float(alpha)
    if alpha not in _Z_FOR_TWO_SIDED:
        raise InvalidConfigurationError(
            f"alpha must be one of {sorted(_Z_FOR_TWO_SIDED)}, got {alpha}."
        )
    return _Z_FOR_TWO_SIDED[alpha]
