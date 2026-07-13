from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CyclicalTestConfig:
    """Configuration shared by the pieces of the cyclical test."""

    n_deterministic_cycles: int = 4  # Number m of Chebyshev terms P_1(t), ..., P_m(t); 0 means none.
    include_intercept: bool = False  # Include P_0(t)=1 in the regression.
    chebyshev_orders: Optional[Tuple[int, ...]] = None  # Explicit positive Chebyshev orders; None means P_1, ..., P_m.
    d_grid: Optional[np.ndarray] = field(default=None)  # Fixed-grid D values; only used when d_search_strategy="fixed_grid".
    d_search_strategy: str = "adaptive"  # "adaptive" coarse-to-fine search or "fixed_grid".
    d_coarse_grid: Optional[np.ndarray] = field(default=None)  # Adaptive coarse D grid; None means [0.0, 0.1, ..., 1.0].
    d_fine_step: float = 0.01  # Step of the local fine grid in adaptive search.
    d_fine_radius: float = 0.09  # Half-width of the local fine grid around the best coarse D.
    r_window: int = 10  # Half-window around the periodogram peak R*.
    top_k: int = 1  # Number of best candidates to keep.
    variance_mode: str = "time"  # Variance estimator: "time", "frequency", or "both".
    statistic_mode: str = "test"  # Statistic formula: "test" or "test_star".
    stochastic_cycle_mode: str = "single"  # Cycle search mode used by the grid.
    n_stochastic_cycles: int = 1  # Number of periodogram peaks used when stochastic_cycle_mode="multi_cycle".
    ignored_stochastic_rs: Optional[Tuple[int, ...]] = None  # Frequency indices excluded from stochastic-cycle search.
    error_model: str = "white_noise"  # Residual error specification: white noise, AR(1), or AR(2).
    drop_singular_frequency: bool = True  # Drop j=R where psi is singular.
    exclude_zero_frequency: bool = True  # Ignore zero frequency when locating R*.
    return_residuals_for_top_k: bool = True  # Store residuals for retained candidates.
    return_residuals_for_threshold: bool = False  # Store residuals for candidates retained only by threshold.
