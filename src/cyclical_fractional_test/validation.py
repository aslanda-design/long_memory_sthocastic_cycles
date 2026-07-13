from __future__ import annotations

from typing import Any, Collection, Optional, Union

import numpy as np

from .exceptions import InvalidConfigurationError, InvalidCycleError, InvalidSeriesError

_VARIANCE_MODES = {"time", "frequency", "both"}
_STATISTIC_MODES = {"test", "test_star"}
_STOCHASTIC_CYCLE_MODES = {"single", "multi_peak_single_cycle", "multi_cycle"}
_ERROR_MODELS = {"white_noise", "ar1", "ar2"}
_D_SEARCH_STRATEGIES = {"adaptive", "fixed_grid"}

_MIN_SERIES_LENGTH = 5


def validate_series(y: Any, min_length: int = _MIN_SERIES_LENGTH) -> np.ndarray:
    """Turn Y(t) into a clean one-dimensional float array.

    Invalid inputs raise InvalidSeriesError with the first problem found.
    """
    if y is None:
        raise InvalidSeriesError("Series y must not be None.")

    try:
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidSeriesError(
            f"Series y cannot be converted to a numeric array: {exc}"
        ) from exc

    if arr.ndim != 1:
        raise InvalidSeriesError(
            f"Series y must be 1-dimensional, got shape {arr.shape}."
        )
    if arr.size == 0:
        raise InvalidSeriesError("Series y must not be empty.")
    if arr.size < min_length:
        raise InvalidSeriesError(
            f"Series y has {arr.size} observations; at least {min_length} are required."
        )
    if np.any(np.isnan(arr)):
        raise InvalidSeriesError("Series y contains NaN values.")
    if np.any(np.isinf(arr)):
        raise InvalidSeriesError("Series y contains infinite values.")

    return arr


def validate_n_deterministic_cycles(n: Any) -> int:
    """Check the requested number of deterministic Chebyshev terms."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise InvalidConfigurationError(
            f"n_deterministic_cycles must be an int, got {type(n).__name__}."
        )
    if n < 0:
        raise InvalidConfigurationError(
            f"n_deterministic_cycles must be >= 0, got {n}."
        )
    return n


def validate_chebyshev_orders(orders: Any) -> Optional[tuple[int, ...]]:
    """Check explicit deterministic Chebyshev orders, if provided."""
    if orders is None:
        return None
    try:
        order_list = list(orders)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"chebyshev_orders must be an iterable of positive ints, got "
            f"{type(orders).__name__}."
        ) from exc
    if len(order_list) == 0:
        raise InvalidConfigurationError("chebyshev_orders must not be empty.")

    validated: list[int] = []
    for order in order_list:
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise InvalidConfigurationError(
                f"chebyshev_orders must contain positive ints, got {order!r}."
            )
        order_int = int(order)
        if order_int <= 0:
            raise InvalidConfigurationError(
                "chebyshev_orders must contain positive orders only; "
                "use include_intercept=True for P_0."
            )
        validated.append(order_int)
    if len(set(validated)) != len(validated):
        raise InvalidConfigurationError("chebyshev_orders must not contain duplicates.")
    return tuple(validated)


def validate_n_stochastic_cycles(n: Any) -> int:
    """Check the requested number of simultaneous stochastic cycles."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise InvalidConfigurationError(
            f"n_stochastic_cycles must be an int, got {type(n).__name__}."
        )
    if n < 1:
        raise InvalidConfigurationError(
            f"n_stochastic_cycles must be >= 1, got {n}."
        )
    return n


def validate_ignored_stochastic_rs(values: Any) -> Optional[tuple[int, ...]]:
    """Check stochastic frequency indices excluded from the search."""
    if values is None:
        return None
    try:
        value_list = list(values)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"ignored_stochastic_rs must be an iterable of non-negative ints, got "
            f"{type(values).__name__}."
        ) from exc
    if len(value_list) == 0:
        raise InvalidConfigurationError("ignored_stochastic_rs must not be empty.")

    validated: list[int] = []
    for value in value_list:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise InvalidConfigurationError(
                f"ignored_stochastic_rs must contain ints, got {value!r}."
            )
        value_int = int(value)
        if value_int < 0:
            raise InvalidConfigurationError(
                f"ignored_stochastic_rs must be >= 0, got {value_int}."
            )
        validated.append(value_int)
    if len(set(validated)) != len(validated):
        raise InvalidConfigurationError(
            "ignored_stochastic_rs must not contain duplicates."
        )
    return tuple(validated)


def validate_top_k(top_k: Any) -> int:
    """Check how many ranked candidates should be kept."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise InvalidConfigurationError(
            f"top_k must be an int, got {type(top_k).__name__}."
        )
    if top_k < 1:
        raise InvalidConfigurationError(f"top_k must be >= 1, got {top_k}.")
    return top_k


def validate_r_window(r_window: Any) -> int:
    """Check the half-width used around the periodogram peak."""
    if not isinstance(r_window, int) or isinstance(r_window, bool):
        raise InvalidConfigurationError(
            f"r_window must be an int, got {type(r_window).__name__}."
        )
    if r_window < 0:
        raise InvalidConfigurationError(f"r_window must be >= 0, got {r_window}.")
    return r_window


def validate_d_grid(d_grid: Any) -> Optional[np.ndarray]:
    """Check the candidate grid for D.

    None is left untouched so the default grid can be built at runtime.
    User grids must be one-dimensional numeric values in [0, 1].
    """
    if d_grid is None:
        return None

    try:
        arr = np.asarray(d_grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"d_grid cannot be converted to a numeric array: {exc}"
        ) from exc

    if arr.ndim != 1:
        raise InvalidConfigurationError(
            f"d_grid must be 1-dimensional, got shape {arr.shape}."
        )
    if arr.size == 0:
        raise InvalidConfigurationError("d_grid must not be empty.")
    if np.any(np.isnan(arr)):
        raise InvalidConfigurationError("d_grid contains NaN values.")
    if np.any(np.isinf(arr)):
        raise InvalidConfigurationError("d_grid contains infinite values.")
    if np.any(arr < 0.0):
        raise InvalidConfigurationError(
            "All d_grid values must be >= 0.0 for the base test."
        )
    if np.any(arr > 1.0):
        raise InvalidConfigurationError(
            "All d_grid values must be <= 1.0 for the base test."
        )

    return arr


def validate_d_coarse_grid(d_coarse_grid: Any) -> Optional[np.ndarray]:
    """Check the adaptive coarse grid for D.

    None is left untouched so the default coarse grid is built at runtime.
    User grids must be one-dimensional numeric values in [0, 1].
    """
    if d_coarse_grid is None:
        return None

    try:
        arr = np.asarray(d_coarse_grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"d_coarse_grid cannot be converted to a numeric array: {exc}"
        ) from exc

    if arr.ndim != 1:
        raise InvalidConfigurationError(
            f"d_coarse_grid must be 1-dimensional, got shape {arr.shape}."
        )
    if arr.size == 0:
        raise InvalidConfigurationError("d_coarse_grid must not be empty.")
    if np.any(np.isnan(arr)):
        raise InvalidConfigurationError("d_coarse_grid contains NaN values.")
    if np.any(np.isinf(arr)):
        raise InvalidConfigurationError("d_coarse_grid contains infinite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise InvalidConfigurationError(
            "All d_coarse_grid values must lie in [0.0, 1.0]."
        )

    return arr


def validate_d_fine_step(d_fine_step: Any) -> float:
    """Check the fine-grid step used in adaptive search; must be positive and finite."""
    if isinstance(d_fine_step, bool):
        raise InvalidConfigurationError("d_fine_step must be numeric, got bool.")
    try:
        value = float(d_fine_step)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"d_fine_step must be numeric, got {type(d_fine_step).__name__}."
        ) from exc
    if not np.isfinite(value):
        raise InvalidConfigurationError(f"d_fine_step must be finite, got {d_fine_step!r}.")
    if value <= 0.0:
        raise InvalidConfigurationError(f"d_fine_step must be > 0, got {value}.")
    return value


def validate_d_fine_radius(d_fine_radius: Any) -> float:
    """Check the fine-grid half-width used in adaptive search; must be positive and finite."""
    if isinstance(d_fine_radius, bool):
        raise InvalidConfigurationError("d_fine_radius must be numeric, got bool.")
    try:
        value = float(d_fine_radius)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"d_fine_radius must be numeric, got {type(d_fine_radius).__name__}."
        ) from exc
    if not np.isfinite(value):
        raise InvalidConfigurationError(f"d_fine_radius must be finite, got {d_fine_radius!r}.")
    if value <= 0.0:
        raise InvalidConfigurationError(f"d_fine_radius must be > 0, got {value}.")
    return value


def validate_mode(value: Any, allowed_values: Collection[str], field_name: str) -> str:
    """Check a string option against its accepted values."""
    if not isinstance(value, str):
        raise InvalidConfigurationError(
            f"{field_name} must be a str, got {type(value).__name__}."
        )
    if value not in allowed_values:
        raise InvalidConfigurationError(
            f"{field_name} must be one of {sorted(allowed_values)}, got {value!r}."
        )
    return value


def validate_boolean(value: Any, field_name: str) -> bool:
    """Check options that must be plain Python booleans."""
    if not isinstance(value, bool):
        raise InvalidConfigurationError(
            f"{field_name} must be a bool, got {type(value).__name__} ({value!r})."
        )
    return value


def validate_cycle(cycle: Any, T: Optional[int] = None) -> "StochasticCycle":
    """Check one StochasticCycle before it enters the numerical routines.

    When T is known, R is also checked against the series length.
    """
    from .results import StochasticCycle  # local import avoids circular dependency

    if not isinstance(cycle, StochasticCycle):
        raise InvalidCycleError(
            f"Expected a StochasticCycle instance, got {type(cycle).__name__}."
        )

    R, D = cycle.R, cycle.D

    if not isinstance(R, int) or isinstance(R, bool):
        raise InvalidCycleError(
            f"StochasticCycle.R must be an int, got {type(R).__name__} ({R!r})."
        )
    if R < 0:
        raise InvalidCycleError(
            f"StochasticCycle.R must be >= 0, got {R}."
        )
    if T is not None and R >= T:
        raise InvalidCycleError(
            f"StochasticCycle.R must be < T={T}, got R={R}."
        )

    try:
        D_float = float(D)
    except (TypeError, ValueError) as exc:
        raise InvalidCycleError(
            f"StochasticCycle.D must be numeric, got {type(D).__name__} ({D!r})."
        ) from exc

    if not np.isfinite(D_float):
        raise InvalidCycleError(
            f"StochasticCycle.D must be finite, got {D}."
        )
    if D_float < 0.0:
        raise InvalidCycleError(
            f"StochasticCycle.D must be >= 0.0, got {D}."
        )
    if D_float > 1.0:
        raise InvalidCycleError(
            f"StochasticCycle.D must be <= 1.0, got {D}."
        )

    return cycle


def validate_cycles(
    cycles: Any,
    T: Optional[int] = None,
    allow_multi_cycle: bool = True,
) -> "tuple[StochasticCycle, ...]":
    """Check a cycle collection and return it as a tuple."""
    from .results import StochasticCycle  # local import avoids circular dependency

    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidCycleError(
            f"cycles must be an iterable of StochasticCycle, got {type(cycles).__name__}."
        ) from exc

    if len(cycle_list) == 0:
        raise InvalidCycleError("cycles must not be empty.")

    if not allow_multi_cycle and len(cycle_list) != 1:
        raise InvalidCycleError(
            f"allow_multi_cycle=False requires exactly 1 cycle, got {len(cycle_list)}."
        )

    validated = tuple(validate_cycle(c, T=T) for c in cycle_list)
    return validated


def validate_config(config: Any) -> "CyclicalTestConfig":
    """Check a CyclicalTestConfig field by field.

    The original object is returned when every field is valid.
    """
    from .config import CyclicalTestConfig  # local import avoids circular dependency

    if not isinstance(config, CyclicalTestConfig):
        raise InvalidConfigurationError(
            f"config must be a CyclicalTestConfig instance, got {type(config).__name__}."
        )

    validate_n_deterministic_cycles(config.n_deterministic_cycles)
    validate_boolean(config.include_intercept, "include_intercept")
    validate_chebyshev_orders(config.chebyshev_orders)
    validate_d_grid(config.d_grid)
    validate_mode(config.d_search_strategy, _D_SEARCH_STRATEGIES, "d_search_strategy")
    validate_d_coarse_grid(config.d_coarse_grid)
    validate_d_fine_step(config.d_fine_step)
    validate_d_fine_radius(config.d_fine_radius)
    validate_r_window(config.r_window)
    validate_top_k(config.top_k)
    validate_mode(config.variance_mode, _VARIANCE_MODES, "variance_mode")
    validate_mode(config.statistic_mode, _STATISTIC_MODES, "statistic_mode")
    validate_mode(
        config.stochastic_cycle_mode, _STOCHASTIC_CYCLE_MODES, "stochastic_cycle_mode"
    )
    validate_n_stochastic_cycles(config.n_stochastic_cycles)
    validate_ignored_stochastic_rs(config.ignored_stochastic_rs)
    validate_mode(config.error_model, _ERROR_MODELS, "error_model")
    validate_boolean(config.drop_singular_frequency, "drop_singular_frequency")
    validate_boolean(config.exclude_zero_frequency, "exclude_zero_frequency")
    validate_boolean(config.return_residuals_for_top_k, "return_residuals_for_top_k")
    validate_boolean(
        config.return_residuals_for_threshold, "return_residuals_for_threshold"
    )

    return config
