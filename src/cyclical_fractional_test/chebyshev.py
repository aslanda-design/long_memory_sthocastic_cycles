from __future__ import annotations

import numpy as np

from .exceptions import InvalidConfigurationError


def build_single_chebyshev_polynomial(T: int, order: int) -> np.ndarray:
    """Build the deterministic Chebyshev polynomial P_k(t) for t = 1, ..., T.

    P_0(t) = 1
    P_k(t) = 2 * cos(k * pi * (t - 0.5) / T)   for k >= 1
    """
    _validate_single_polynomial(T, order)

    if order == 0:
        return np.ones(T, dtype=float)
    t = np.arange(1, T + 1, dtype=float)
    return 2.0 * np.cos(order * np.pi * (t - 0.5) / T)


def evaluate_single_chebyshev_polynomial(
    t_values: np.ndarray,
    T_ref: int,
    order: int,
) -> np.ndarray:
    """Evaluate P_k(t) at arbitrary time indices, keeping T_ref fixed.

    Uses the same normalisation P_k(t) = 2 cos(k π (t − 0.5) / T_ref) as the
    in-sample basis, so for t = 1, ..., T_ref this reproduces
    build_single_chebyshev_polynomial. Indices t > T_ref extrapolate the basis.
    """
    _validate_evaluate_polynomial(t_values, T_ref, order)
    t = np.asarray(t_values, dtype=float)
    if order == 0:
        return np.ones_like(t)
    return 2.0 * np.cos(order * np.pi * (t - 0.5) / T_ref)


def build_chebyshev_design(
    T: int,
    n_cycles: int,
    include_intercept: bool = False,
    orders: object | None = None,
) -> np.ndarray:
    """Build the deterministic Chebyshev design matrix.

    Without intercept: columns [P_1, ..., P_n_cycles], shape (T, n_cycles).
    With intercept:    columns [P_0, P_1, ..., P_n_cycles], shape (T, n_cycles+1).
    When n_cycles=0 this returns either no columns or only P_0.
    If orders is given, columns use exactly those positive Chebyshev orders,
    with P_0 prepended only when include_intercept=True.
    Without explicit orders, generates exactly n_cycles columns — no zero-padded extras.
    """
    _validate_chebyshev_design(T, n_cycles, include_intercept, orders)

    resolved_orders = _resolve_chebyshev_orders(n_cycles, include_intercept, orders)
    n_cols = len(resolved_orders)
    X = np.empty((T, n_cols), dtype=float)
    for col_idx, k in enumerate(resolved_orders):
        X[:, col_idx] = build_single_chebyshev_polynomial(T, k)
    return X


def build_chebyshev_design_at(
    t_values: np.ndarray,
    T_ref: int,
    n_cycles: int,
    include_intercept: bool = False,
    orders: object | None = None,
) -> np.ndarray:
    """Build the Chebyshev design matrix at arbitrary time indices.

    Same column layout as build_chebyshev_design but evaluated at t_values with
    the training length T_ref held fixed, so it can extrapolate the basis to
    t > T_ref for out-of-sample prediction. For t_values = 1, ..., T_ref it
    reproduces build_chebyshev_design(T_ref, n_cycles, include_intercept).
    """
    _validate_chebyshev_design_at(t_values, T_ref, n_cycles, include_intercept, orders)

    t_arr = np.asarray(t_values, dtype=float)
    resolved_orders = _resolve_chebyshev_orders(n_cycles, include_intercept, orders)
    n_cols = len(resolved_orders)
    X = np.empty((len(t_arr), n_cols), dtype=float)
    for col_idx, k in enumerate(resolved_orders):
        X[:, col_idx] = evaluate_single_chebyshev_polynomial(t_arr, T_ref, k)
    return X


def _resolve_chebyshev_orders(
    n_cycles: int,
    include_intercept: bool,
    orders: object | None,
) -> tuple[int, ...]:
    """Return the column order for a Chebyshev design matrix."""
    if orders is None:
        start_order = 0 if include_intercept else 1
        return tuple(range(start_order, n_cycles + 1))

    explicit_orders = _validate_explicit_chebyshev_orders(orders)
    if include_intercept:
        return (0,) + explicit_orders
    return explicit_orders


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact 
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_single_polynomial(T: int, order: int) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T <= 0:
        raise InvalidConfigurationError(f"T must be > 0, got {T}.")
    if isinstance(order, bool) or not isinstance(order, int):
        raise InvalidConfigurationError(
            f"order must be an int, got {type(order).__name__}."
        )
    if order < 0:
        raise InvalidConfigurationError(f"order must be >= 0, got {order}.")


def _validate_evaluate_polynomial(t_values: object, T_ref: int, order: int) -> None:
    if isinstance(T_ref, bool) or not isinstance(T_ref, int):
        raise InvalidConfigurationError(
            f"T_ref must be an int, got {type(T_ref).__name__}."
        )
    if T_ref <= 0:
        raise InvalidConfigurationError(f"T_ref must be > 0, got {T_ref}.")
    if isinstance(order, bool) or not isinstance(order, int):
        raise InvalidConfigurationError(
            f"order must be an int, got {type(order).__name__}."
        )
    if order < 0:
        raise InvalidConfigurationError(f"order must be >= 0, got {order}.")
    try:
        t_arr = np.asarray(t_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"t_values must be numeric: {exc}") from exc
    if t_arr.ndim != 1 or t_arr.size == 0:
        raise InvalidConfigurationError("t_values must be a non-empty 1-D array.")


def _validate_chebyshev_design_at(
    t_values: object,
    T_ref: int,
    n_cycles: int,
    include_intercept: bool,
    orders: object | None = None,
) -> None:
    if isinstance(T_ref, bool) or not isinstance(T_ref, int):
        raise InvalidConfigurationError(
            f"T_ref must be an int, got {type(T_ref).__name__}."
        )
    if T_ref <= 0:
        raise InvalidConfigurationError(f"T_ref must be > 0, got {T_ref}.")
    if isinstance(n_cycles, bool) or not isinstance(n_cycles, int):
        raise InvalidConfigurationError(
            f"n_cycles must be an int, got {type(n_cycles).__name__}."
        )
    if n_cycles < 0:
        raise InvalidConfigurationError(f"n_cycles must be >= 0, got {n_cycles}.")
    if not isinstance(include_intercept, bool):
        raise InvalidConfigurationError(
            f"include_intercept must be a bool, got {type(include_intercept).__name__}."
        )
    if orders is not None:
        _validate_explicit_chebyshev_orders(orders)
    try:
        t_arr = np.asarray(t_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"t_values must be numeric: {exc}") from exc
    if t_arr.ndim != 1 or t_arr.size == 0:
        raise InvalidConfigurationError("t_values must be a non-empty 1-D array.")


def _validate_chebyshev_design(
    T: int,
    n_cycles: int,
    include_intercept: bool,
    orders: object | None = None,
) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T <= 0:
        raise InvalidConfigurationError(f"T must be > 0, got {T}.")
    if isinstance(n_cycles, bool) or not isinstance(n_cycles, int):
        raise InvalidConfigurationError(
            f"n_cycles must be an int, got {type(n_cycles).__name__}."
        )
    if n_cycles < 0:
        raise InvalidConfigurationError(f"n_cycles must be >= 0, got {n_cycles}.")
    if not isinstance(include_intercept, bool):
        raise InvalidConfigurationError(
            f"include_intercept must be a bool, got {type(include_intercept).__name__}."
        )
    if orders is not None:
        _validate_explicit_chebyshev_orders(orders)


def _validate_explicit_chebyshev_orders(orders: object) -> tuple[int, ...]:
    try:
        order_list = list(orders)  # type: ignore[arg-type]
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"chebyshev orders must be iterable, got {type(orders).__name__}."
        ) from exc
    if len(order_list) == 0:
        raise InvalidConfigurationError("chebyshev orders must not be empty.")

    validated: list[int] = []
    for order in order_list:
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise InvalidConfigurationError(
                f"chebyshev orders must be positive ints, got {order!r}."
            )
        order_int = int(order)
        if order_int <= 0:
            raise InvalidConfigurationError(
                "chebyshev orders must be positive; use include_intercept=True "
                "to include P_0."
            )
        validated.append(order_int)
    if len(set(validated)) != len(validated):
        raise InvalidConfigurationError("chebyshev orders must not contain duplicates.")
    return tuple(validated)
