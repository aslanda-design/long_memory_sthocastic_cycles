#!/usr/bin/env python3
"""Resumable grid search for the energy-demand time series.

The real grid is intentionally sequential and checkpointed one configuration at
a time because each fit can take a long time. A completed configuration is never
considered resumable until all artifacts have been written and status.json says
"done".
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pickle
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cyclical_fractional_test import (  # noqa: E402
    CyclicalFractionalModel,
    CyclicalTestConfig,
)


DEFAULT_DATA = Path("data/energy_demand.csv")
DEFAULT_COLUMN = "AT_load_actual_entsoe_transparency"
DEFAULT_OUTPUT_DIR = Path("data/gridsearch_runs/energy_at_detrend")
DEFAULT_SMOKE_OUTPUT_DIR = Path("data/gridsearch_runs/energy_at_detrend_smoke")
ERROR_MODELS = ("white_noise", "ar1", "ar2")
DETERMINISTIC_CYCLE_COUNTS = (0, 1, 2, 3)


@dataclass(frozen=True)
class GridRunConfig:
    """One outer-grid configuration."""

    config_id: str
    error_model: str
    n_deterministic_cycles: int


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = build_grid_configs()

    if args.config_id is not None:
        known_ids = {config.config_id for config in configs}
        if args.config_id not in known_ids:
            raise SystemExit(
                f"Unknown --config-id {args.config_id!r}. "
                f"Known ids: {', '.join(sorted(known_ids))}"
            )
        configs = [config for config in configs if config.config_id == args.config_id]

    if args.dry_run:
        print_dry_run(configs)
        return 0

    output_dir = resolve_output_dir(args.output_dir, args.smoke)
    data_path = Path(args.data)
    y, data_info = load_energy_series(data_path, args.column)

    if args.smoke:
        if args.smoke_rows < 30:
            raise SystemExit("--smoke-rows must be at least 30.")
        y = y[: min(args.smoke_rows, len(y))]
        data_info = {
            **data_info,
            "smoke": True,
            "smoke_rows_requested": int(args.smoke_rows),
            "rows_used": int(len(y)),
        }
    else:
        data_info = {**data_info, "smoke": False, "rows_used": int(len(y))}

    train_size = compute_train_size(len(y), args.split_ratio)
    y_train = y[:train_size]
    trend_coefficients = fit_train_linear_trend(y_train)
    trend_full = evaluate_linear_trend(trend_coefficients, len(y))
    y_train_detrended = y_train - trend_full[:train_size]

    run_context = {
        "data_path": str(data_path),
        "column": args.column,
        "split_ratio": float(args.split_ratio),
        "train_size": int(train_size),
        "test_size": int(len(y) - train_size),
        "n_observations": int(len(y)),
        "detrend": "linear trend fitted on train only",
        "trend_coefficients": trend_coefficients.tolist(),
        "created_at": utc_now(),
        **data_info,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0

    for index, outer_config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] {outer_config.config_id}: "
            f"error_model={outer_config.error_model}, "
            f"deterministic_cycles={outer_config.n_deterministic_cycles}",
            flush=True,
        )
        metrics = run_one_configuration(
            outer_config=outer_config,
            y=y,
            y_train_detrended=y_train_detrended,
            trend_full=trend_full,
            train_size=train_size,
            output_dir=output_dir,
            run_context=run_context,
            force=args.force,
            smoke=args.smoke,
            threshold=args.threshold,
            threshold_store_residuals=args.threshold_store_residuals,
        )
        if metrics is None:
            failed += 1
        write_leaderboards(output_dir)

    write_leaderboards(output_dir)
    print(f"Leaderboard written to {output_dir / 'leaderboard.csv'}", flush=True)
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable gridsearch for the energy-demand time series."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV file to read.")
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="Demand column in the CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for artifacts. Defaults to "
            "data/gridsearch_runs/energy_at_detrend, or *_smoke with --smoke."
        ),
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="Chronological train fraction. Default: 0.9.",
    )
    parser.add_argument(
        "--config-id",
        default=None,
        help="Run one config only, e.g. white_noise_det0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a completed artifact already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned configurations without reading data or writing files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast end-to-end check on a short prefix and tiny adaptive D grid.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=240,
        help="Number of initial observations used with --smoke. Default: 240.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Positive statistic threshold for storing all candidates below it. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--threshold-store-residuals",
        action="store_true",
        help=(
            "Keep full residual vectors for every candidate stored by --threshold. "
            "This makes under-threshold candidates prediction-ready, but can create "
            "very large artifacts."
        ),
    )
    return parser.parse_args(argv)


def resolve_output_dir(output_dir: str | None, smoke: bool) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return DEFAULT_SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR


def build_grid_configs() -> list[GridRunConfig]:
    configs: list[GridRunConfig] = []
    for error_model in ERROR_MODELS:
        for n_det in DETERMINISTIC_CYCLE_COUNTS:
            configs.append(
                GridRunConfig(
                    config_id=f"{error_model}_det{n_det}",
                    error_model=error_model,
                    n_deterministic_cycles=n_det,
                )
            )
    return configs


def print_dry_run(configs: Iterable[GridRunConfig]) -> None:
    configs = list(configs)
    print("Planned configurations:")
    for config in configs:
        print(
            f"- {config.config_id}: "
            f"error_model={config.error_model}, "
            f"n_deterministic_cycles={config.n_deterministic_cycles}, "
            "include_intercept=True, stochastic_cycle_mode=multi_cycle, "
            "n_stochastic_cycles=3, exclude_zero_frequency=True"
        )
    print(f"Total configs: {len(configs)}")


def load_energy_series(path: Path, column: str) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[float] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a header row.")
        if column not in reader.fieldnames:
            raise ValueError(
                f"Column {column!r} not found in {path}. "
                f"Available columns include: {reader.fieldnames[:10]}"
            )

        has_timestamp = "utc_timestamp" in reader.fieldnames
        for row_number, row in enumerate(reader, start=2):
            raw_value = row.get(column, "")
            if raw_value is None or raw_value.strip() == "":
                raise ValueError(f"Missing value in {column!r} at CSV row {row_number}.")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in {column!r} at CSV row {row_number}: "
                    f"{raw_value!r}"
                ) from exc
            values.append(value)

            if has_timestamp:
                timestamp = row.get("utc_timestamp")
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp

    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Loaded series is empty.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Loaded series contains NaN or infinite values.")

    info = {
        "source_rows": int(y.size),
        "first_utc_timestamp": first_timestamp,
        "last_utc_timestamp": last_timestamp,
        "raw_min": float(np.min(y)),
        "raw_max": float(np.max(y)),
        "raw_mean": float(np.mean(y)),
    }
    return y, info


def compute_train_size(n_observations: int, split_ratio: float) -> int:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"--split-ratio must be in (0, 1), got {split_ratio}.")
    train_size = int(n_observations * split_ratio)
    if train_size < 5:
        raise ValueError(
            f"Train split has only {train_size} observations; at least 5 are required."
        )
    if train_size >= n_observations:
        raise ValueError("Train split leaves no holdout observations.")
    return train_size


def fit_train_linear_trend(y_train: np.ndarray) -> np.ndarray:
    """Fit y_t = slope * t + intercept using train observations only."""
    t_train = np.arange(len(y_train), dtype=float)
    return np.polyfit(t_train, np.asarray(y_train, dtype=float), deg=1)


def evaluate_linear_trend(coefficients: np.ndarray, n_observations: int) -> np.ndarray:
    t = np.arange(n_observations, dtype=float)
    return np.polyval(coefficients, t)


def run_one_configuration(
    *,
    outer_config: GridRunConfig,
    y: np.ndarray,
    y_train_detrended: np.ndarray,
    trend_full: np.ndarray,
    train_size: int,
    output_dir: Path,
    run_context: dict[str, Any],
    force: bool,
    smoke: bool,
    threshold: float | None,
    threshold_store_residuals: bool = False,
) -> dict[str, Any] | None:
    run_dir = output_dir / outer_config.config_id
    artifact_path = run_dir / "artifact.pkl"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"

    if not force and is_completed(run_dir):
        print(f"  skip: completed artifact already exists at {artifact_path}", flush=True)
        return read_json(metrics_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_model_config(
        outer_config,
        smoke=smoke,
        threshold_store_residuals=threshold_store_residuals,
    )
    atomic_write_json(run_dir / "config.json", config_to_json(config))
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "config_id": outer_config.config_id,
            "started_at": utc_now(),
            "artifact_path": str(artifact_path),
        },
    )

    started = time.perf_counter()
    started_at = utc_now()
    try:
        model = CyclicalFractionalModel(config=config, threshold=threshold).fit(
            y_train_detrended
        )
        prediction_detrended = model.predict(len(y))
        prediction = prediction_detrended + trend_full
        errors = y[train_size:] - prediction[train_size:]
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        elapsed_seconds = float(time.perf_counter() - started)

        metrics = build_metrics(
            outer_config=outer_config,
            model=model,
            rmse=rmse,
            mae=mae,
            elapsed_seconds=elapsed_seconds,
            train_size=train_size,
            total_size=len(y),
            started_at=started_at,
            completed_at=utc_now(),
            smoke=smoke,
        )
        metadata = {
            **run_context,
            "config_id": outer_config.config_id,
            "started_at": started_at,
            "completed_at": metrics["completed_at"],
            "elapsed_seconds": elapsed_seconds,
            "model_class": "CyclicalFractionalModel",
        }
        artifact = {
            "model": model,
            "config": config,
            "trend_coefficients": np.asarray(run_context["trend_coefficients"]),
            "metrics": metrics,
            "metadata": metadata,
        }

        atomic_write_pickle(artifact_path, artifact)
        atomic_write_npz(
            run_dir / "predictions.npz",
            y_observed=y,
            trend=trend_full,
            y_detrended_train=y_train_detrended,
            prediction_detrended=prediction_detrended,
            prediction=prediction,
            train_size=np.asarray(train_size, dtype=np.int64),
        )
        atomic_write_json(run_dir / "top_k.json", summarize_top_k(model, train_size))
        atomic_write_json(metrics_path, metrics)
        atomic_write_json(
            status_path,
            {
                "status": "done",
                "config_id": outer_config.config_id,
                "started_at": started_at,
                "completed_at": metrics["completed_at"],
                "elapsed_seconds": elapsed_seconds,
                "artifact_path": str(artifact_path),
            },
        )
        print(
            f"  done: RMSE={rmse:.6g}, MAE={mae:.6g}, "
            f"elapsed={elapsed_seconds / 60.0:.2f} min",
            flush=True,
        )
        return metrics
    except Exception as exc:  # keep the search resumable across bad configurations
        elapsed_seconds = float(time.perf_counter() - started)
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "config_id": outer_config.config_id,
                "started_at": started_at,
                "failed_at": utc_now(),
                "elapsed_seconds": elapsed_seconds,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"  failed: {exc!r}", flush=True)
        return None


def is_completed(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    artifact_path = run_dir / "artifact.pkl"
    if not status_path.exists() or not artifact_path.exists():
        return False
    try:
        status = read_json(status_path)
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "done"


def build_model_config(
    outer_config: GridRunConfig,
    *,
    smoke: bool,
    threshold_store_residuals: bool = False,
) -> CyclicalTestConfig:
    common = dict(
        n_deterministic_cycles=outer_config.n_deterministic_cycles,
        include_intercept=True,
        top_k=5,
        statistic_mode="test",
        stochastic_cycle_mode="multi_cycle",
        n_stochastic_cycles=3,
        exclude_zero_frequency=True,
        error_model=outer_config.error_model,
        d_search_strategy="adaptive",
        return_residuals_for_threshold=threshold_store_residuals,
    )
    if smoke:
        return CyclicalTestConfig(
            **common,
            d_coarse_grid=np.array([0.0, 0.5, 1.0]),
            d_fine_step=0.25,
            d_fine_radius=0.25,
        )
    return CyclicalTestConfig(**common)


def build_metrics(
    *,
    outer_config: GridRunConfig,
    model: CyclicalFractionalModel,
    rmse: float,
    mae: float,
    elapsed_seconds: float,
    train_size: int,
    total_size: int,
    started_at: str,
    completed_at: str,
    smoke: bool,
) -> dict[str, Any]:
    result = model.result_
    best = result.best_result
    if best is None:
        raise RuntimeError("Fitted model did not expose a best_result.")
    cycles = summarize_cycles(best.cycles, train_size)
    return {
        "config_id": outer_config.config_id,
        "error_model": outer_config.error_model,
        "n_deterministic_cycles": int(outer_config.n_deterministic_cycles),
        "include_intercept": True,
        "stochastic_cycle_mode": "multi_cycle",
        "n_stochastic_cycles": 3,
        "exclude_zero_frequency": True,
        "d_search_strategy": model.config.d_search_strategy,
        "threshold": none_or_float(model.threshold),
        "n_under_threshold_candidates": count_under_threshold_candidates(model),
        "under_threshold_frequency_keys": under_threshold_frequency_keys(model),
        "rmse": rmse,
        "mae": mae,
        "test_value": none_or_float(best.test_value),
        "test_star_value": none_or_float(best.test_star_value),
        "score_abs_test": none_or_float(best.abs_test_value),
        "cycles": cycles,
        "ar_coefficients": [float(value) for value in best.ar_coefficients],
        "n_candidates_evaluated": none_or_int(result.n_candidates_evaluated),
        "r_candidates": array_to_int_list(result.r_candidates),
        "d_grid": array_to_float_list(result.d_grid),
        "train_size": int(train_size),
        "test_size": int(total_size - train_size),
        "n_observations": int(total_size),
        "elapsed_seconds": elapsed_seconds,
        "started_at": started_at,
        "completed_at": completed_at,
        "smoke": bool(smoke),
    }


def summarize_top_k(model: CyclicalFractionalModel, train_size: int) -> list[dict[str, Any]]:
    top_k = getattr(model.result_, "top_k_results", [])
    return [
        {
            "rank": rank,
            "cycles": summarize_cycles(candidate.cycles, train_size),
            "error_model": candidate.error_model,
            "ar_coefficients": [float(value) for value in candidate.ar_coefficients],
            "test_value": none_or_float(candidate.test_value),
            "test_star_value": none_or_float(candidate.test_star_value),
            "abs_test_value": none_or_float(candidate.abs_test_value),
            "abs_test_star_value": none_or_float(candidate.abs_test_star_value),
            "xa": none_or_float(candidate.xa),
            "xaa": none_or_float(candidate.xaa),
            "variance_time": none_or_float(candidate.variance_time),
            "variance_frequency": none_or_float(candidate.variance_frequency),
            "residual_sum_squares": none_or_float(candidate.residual_sum_squares),
        }
        for rank, candidate in enumerate(top_k, start=1)
    ]


def summarize_cycles(cycles: Iterable[Any], train_size: int) -> list[dict[str, Any]]:
    summary = []
    for cycle in cycles:
        R = int(cycle.R)
        period_hours = float("inf") if R == 0 else float(train_size / R)
        summary.append(
            {
                "R": R,
                "D": float(cycle.D),
                "period_hours": period_hours,
            }
        )
    return summary


def write_leaderboards(output_dir: Path) -> None:
    metrics = collect_completed_metrics(output_dir)
    atomic_write_json(output_dir / "leaderboard.json", metrics)
    rows = [leaderboard_row(item) for item in metrics]
    fieldnames = [
        "rank",
        "config_id",
        "error_model",
        "n_deterministic_cycles",
        "rmse",
        "mae",
        "test_value",
        "test_star_value",
        "cycles",
        "period_hours",
        "ar_coefficients",
        "threshold",
        "n_under_threshold_candidates",
        "n_candidates_evaluated",
        "elapsed_seconds",
        "completed_at",
        "smoke",
    ]
    lines = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.seek(0)
        lines = handle.readlines()
    atomic_write_text(output_dir / "leaderboard.csv", "".join(lines))


def collect_completed_metrics(output_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not output_dir.exists():
        return items
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if not is_completed(run_dir):
            continue
        try:
            items.append(read_json(metrics_path))
        except (OSError, json.JSONDecodeError):
            continue
    items.sort(key=lambda item: (float(item.get("rmse", np.inf)), item["config_id"]))
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    return items


def leaderboard_row(metrics: dict[str, Any]) -> dict[str, Any]:
    cycles = metrics.get("cycles", [])
    return {
        "rank": metrics.get("rank"),
        "config_id": metrics.get("config_id"),
        "error_model": metrics.get("error_model"),
        "n_deterministic_cycles": metrics.get("n_deterministic_cycles"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "test_value": metrics.get("test_value"),
        "test_star_value": metrics.get("test_star_value"),
        "cycles": json.dumps(
            [{"R": cycle["R"], "D": cycle["D"]} for cycle in cycles],
            separators=(",", ":"),
        ),
        "period_hours": json.dumps(
            [cycle["period_hours"] for cycle in cycles],
            separators=(",", ":"),
        ),
        "ar_coefficients": json.dumps(
            metrics.get("ar_coefficients", []),
            separators=(",", ":"),
        ),
        "threshold": metrics.get("threshold"),
        "n_under_threshold_candidates": metrics.get("n_under_threshold_candidates"),
        "n_candidates_evaluated": metrics.get("n_candidates_evaluated"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "completed_at": metrics.get("completed_at"),
        "smoke": metrics.get("smoke"),
    }


def count_under_threshold_candidates(model: CyclicalFractionalModel) -> int:
    grouped = getattr(model.result_, "under_threshold_results", None)
    if not grouped:
        return 0
    return sum(len(candidates) for candidates in grouped.values())


def under_threshold_frequency_keys(model: CyclicalFractionalModel) -> list[list[int]]:
    grouped = getattr(model.result_, "under_threshold_results", None)
    if not grouped:
        return []
    return [[int(value) for value in key] for key in sorted(grouped.keys())]


def config_to_json(config: CyclicalTestConfig) -> dict[str, Any]:
    return sanitize_for_json(dataclasses.asdict(config))


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(sanitize_for_json(payload), indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def none_or_int(value: Any) -> int | None:
    return None if value is None else int(value)


def array_to_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in np.asarray(value).ravel()]


def array_to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value).ravel()]


if __name__ == "__main__":
    raise SystemExit(main())
