from __future__ import annotations

from typing import List, Optional

import numpy as np

from .exceptions import InvalidConfigurationError


# Modes that score by abs(TEST).
_STANDARD_MODES = {"standard", "test"}
# Modes that score by abs(TEST*).
_FREQUENCY_MODES = {"frequency", "test_star"}
_VALID_STATISTIC_MODES = _STANDARD_MODES | _FREQUENCY_MODES


def compute_test_statistic(
    T: int,
    xa: float,
    xaa: float,
    variance_time: float,
) -> float:
    """Compute TEST(R,D) = sqrt(T) / sqrt(XAA) * XA / VAR.

    Sign of XA is preserved; the absolute value is applied only by scoring code.
    """
    _validate_compute_test_statistic(T, xa, xaa, variance_time)

    return float(np.sqrt(T) / np.sqrt(xaa) * xa / variance_time)


def compute_test_star_statistic(
    T: int,
    xa: float,
    xaa: float,
    variance_frequency: float,
) -> float:
    """Compute TEST*(R,D) = sqrt(T) / sqrt(XAA) * XA / VAR*.

    Sign of XA is preserved; the absolute value is applied only by scoring code.
    """
    _validate_compute_test_star_statistic(T, xa, xaa, variance_frequency)
    
    return float(np.sqrt(T) / np.sqrt(xaa) * xa / variance_frequency)


def score_candidate(
    candidate: object,
    statistic_mode: str = "standard",
) -> float:
    """Return |TEST| or |TEST*| of an evaluated candidate; smaller = better.

    Reads test_value / test_star_value already stored on the candidate;
    does not recompute the statistic.
    """
    if statistic_mode not in _VALID_STATISTIC_MODES:
        raise InvalidConfigurationError(
            f"Unknown statistic_mode: {statistic_mode!r}. "
            f"Expected one of {sorted(_VALID_STATISTIC_MODES)}."
        )
    if statistic_mode in _STANDARD_MODES:
        value = getattr(candidate, "test_value", None)
        attr_name = "test_value"
    else:
        value = getattr(candidate, "test_star_value", None)
        attr_name = "test_star_value"
    if value is None:
        raise InvalidConfigurationError(
            f"candidate.{attr_name} is None; cannot score for "
            f"statistic_mode={statistic_mode!r}."
        )
    value_float = float(value)
    if not np.isfinite(value_float):
        raise InvalidConfigurationError(
            f"candidate.{attr_name} is not finite: {value_float}."
        )
    return abs(value_float)


class TopKSelector:
    """Keep only the k candidates with smallest |TEST| (or |TEST*|).

    Ordering is by score_candidate; smaller score is better (closest to zero).
    """

    def __init__(self, k: int, statistic_mode: str = "standard") -> None:
        _validate_top_k_selector_init(k, statistic_mode)
        self._k = k
        self._statistic_mode = statistic_mode
        self._entries: List[tuple] = []  # (score, insertion_index, candidate)
        self._counter = 0

    @property
    def k(self) -> int:
        return self._k

    @property
    def statistic_mode(self) -> str:
        return self._statistic_mode

    def consider(self, candidate: object) -> None:
        """Insert candidate if it is among the k best so far."""
        score = score_candidate(candidate, self._statistic_mode)
        self._entries.append((score, self._counter, candidate))
        self._counter += 1
        self._entries.sort(key=lambda entry: (entry[0], entry[1]))
        if len(self._entries) > self._k:
            self._entries = self._entries[: self._k]

    def get_top_k(self) -> list:
        """Return the retained candidates sorted from best (closest to 0) to worst."""
        return [entry[2] for entry in self._entries]

    def get_best(self) -> Optional[object]:
        """Return the single best candidate, or None if no candidate has been considered."""
        if not self._entries:
            return None
        return self._entries[0][2]


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know the exact
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_positive_T(T: object) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T <= 0:
        raise InvalidConfigurationError(f"T must be > 0, got {T}.")


def _validate_finite_scalar(value: object, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"{name} must be numeric: {exc}") from exc
    if not np.isfinite(v):
        raise InvalidConfigurationError(f"{name} must be finite, got {value!r}.")
    return v


def _validate_positive_scalar(value: object, name: str) -> float:
    v = _validate_finite_scalar(value, name)
    if v <= 0.0:
        raise InvalidConfigurationError(f"{name} must be > 0, got {v}.")
    return v


def _validate_compute_test_statistic(
    T: object, xa: object, xaa: object, variance_time: object
) -> None:
    _validate_positive_T(T)
    _validate_finite_scalar(xa, "xa")
    _validate_positive_scalar(xaa, "xaa")
    _validate_positive_scalar(variance_time, "variance_time")


def _validate_compute_test_star_statistic(
    T: object, xa: object, xaa: object, variance_frequency: object
) -> None:
    _validate_positive_T(T)
    _validate_finite_scalar(xa, "xa")
    _validate_positive_scalar(xaa, "xaa")
    _validate_positive_scalar(variance_frequency, "variance_frequency")


def _validate_top_k_selector_init(k: object, statistic_mode: object) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise InvalidConfigurationError(f"k must be an int, got {type(k).__name__}.")
    if k < 1:
        raise InvalidConfigurationError(f"k must be >= 1, got {k}.")
    if not isinstance(statistic_mode, str):
        raise InvalidConfigurationError(
            f"statistic_mode must be a str, got {type(statistic_mode).__name__}."
        )
    if statistic_mode not in _VALID_STATISTIC_MODES:
        raise InvalidConfigurationError(
            f"Unknown statistic_mode: {statistic_mode!r}. "
            f"Expected one of {sorted(_VALID_STATISTIC_MODES)}."
        )
