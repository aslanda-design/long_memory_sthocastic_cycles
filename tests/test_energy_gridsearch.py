from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "energy_gridsearch.py"
SPEC = importlib.util.spec_from_file_location("energy_gridsearch", SCRIPT_PATH)
energy_gridsearch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = energy_gridsearch
SPEC.loader.exec_module(energy_gridsearch)


class _FakeGridsearchModel:
    last_threshold = None

    def __init__(self, config=None, threshold=None, **kwargs):
        self.config = config
        self.threshold = threshold
        type(self).last_threshold = threshold

    def fit(self, y):
        candidate = SimpleNamespace(
            cycles=(SimpleNamespace(R=1, D=0.25),),
            error_model=self.config.error_model,
            ar_coefficients=(),
            test_value=0.1,
            test_star_value=0.2,
            abs_test_value=0.1,
            abs_test_star_value=0.2,
            xa=0.3,
            xaa=0.4,
            variance_time=0.5,
            variance_frequency=0.6,
            residual_sum_squares=0.7,
        )
        self.result_ = SimpleNamespace(
            best_result=candidate,
            top_k_results=[candidate],
            n_candidates_evaluated=1,
            r_candidates=np.array([1]),
            d_grid=np.array([0.25]),
            under_threshold_results={(1,): [candidate]},
        )
        return self

    def predict(self, n):
        return np.zeros(n, dtype=float)


def test_build_grid_configs_has_expected_outer_grid():
    configs = energy_gridsearch.build_grid_configs()

    assert len(configs) == 12
    assert configs[0].config_id == "white_noise_det0"
    assert configs[-1].config_id == "ar2_det3"
    assert {(c.error_model, c.n_deterministic_cycles) for c in configs} == {
        (error_model, n_det)
        for error_model in ("white_noise", "ar1", "ar2")
        for n_det in (0, 1, 2, 3)
    }


def test_build_model_config_stores_threshold_candidates_lightly():
    outer_config = energy_gridsearch.GridRunConfig(
        config_id="white_noise_det0",
        error_model="white_noise",
        n_deterministic_cycles=0,
    )

    config = energy_gridsearch.build_model_config(outer_config, smoke=False)

    assert config.return_residuals_for_threshold is False


def test_build_model_config_can_store_full_threshold_candidates():
    outer_config = energy_gridsearch.GridRunConfig(
        config_id="white_noise_det0",
        error_model="white_noise",
        n_deterministic_cycles=0,
    )

    config = energy_gridsearch.build_model_config(
        outer_config,
        smoke=False,
        threshold_store_residuals=True,
    )

    assert config.return_residuals_for_threshold is True


def test_linear_trend_is_fit_from_train_only_and_extrapolated():
    y_train = 3.0 + 2.0 * np.arange(10, dtype=float)

    coefficients = energy_gridsearch.fit_train_linear_trend(y_train)
    trend = energy_gridsearch.evaluate_linear_trend(coefficients, 14)

    np.testing.assert_allclose(coefficients, np.array([2.0, 3.0]))
    np.testing.assert_allclose(trend[:10], y_train)
    np.testing.assert_allclose(trend[10:], np.array([23.0, 25.0, 27.0, 29.0]))


def test_smoke_default_output_dir_is_separate_from_real_run():
    assert energy_gridsearch.resolve_output_dir(None, smoke=False).name == (
        "energy_at_detrend"
    )
    assert energy_gridsearch.resolve_output_dir(None, smoke=True).name == (
        "energy_at_detrend_smoke"
    )


def test_parse_args_accepts_threshold():
    args = energy_gridsearch.parse_args(["--dry-run", "--threshold", "2.5"])

    assert args.threshold == 2.5
    assert args.threshold_store_residuals is False


def test_parse_args_accepts_threshold_store_residuals():
    args = energy_gridsearch.parse_args(
        ["--dry-run", "--threshold", "2.5", "--threshold-store-residuals"]
    )

    assert args.threshold_store_residuals is True


def test_run_one_configuration_passes_threshold_and_records_metrics(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        energy_gridsearch, "CyclicalFractionalModel", _FakeGridsearchModel
    )
    outer_config = energy_gridsearch.GridRunConfig(
        config_id="white_noise_det0",
        error_model="white_noise",
        n_deterministic_cycles=0,
    )
    y = np.arange(8, dtype=float)
    y_train_detrended = y[:5]
    trend_full = np.zeros_like(y)
    run_context = {
        "trend_coefficients": np.array([0.0, 0.0]),
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    metrics = energy_gridsearch.run_one_configuration(
        outer_config=outer_config,
        y=y,
        y_train_detrended=y_train_detrended,
        trend_full=trend_full,
        train_size=5,
        output_dir=tmp_path,
        run_context=run_context,
        force=True,
        smoke=True,
        threshold=2.5,
    )

    assert _FakeGridsearchModel.last_threshold == 2.5
    assert metrics["threshold"] == 2.5
    assert metrics["n_under_threshold_candidates"] == 1
    assert metrics["under_threshold_frequency_keys"] == [[1]]
