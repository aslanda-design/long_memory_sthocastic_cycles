# cyclical-fractional-test

`cyclical-fractional-test` is a research-oriented Python package for detecting
fractional cyclic long memory in time series.

The package implements the full test pipeline around candidate cyclic
frequencies `R` and fractional parameters `D`, including deterministic
Chebyshev bases, fractional cyclic filters, candidate scoring, diagnostics,
optional AR residual adjustments, and a small estimator-style wrapper for
prediction.

## Features

- Periodogram and autocorrelogram helpers for exploratory analysis.
- Chebyshev deterministic design matrices with contiguous or explicit orders.
- Single-cycle and aggregate multi-cycle stochastic memory candidates.
- Adaptive coarse-to-fine search over `D`, plus fixed-grid search when exact
  Cartesian evaluation is preferred.
- White-noise, AR(1), and AR(2) residual error specifications.
- `TEST` and `TEST*` statistics with top-k candidate ranking.
- Diagnostics for the periodogram, search grid, variance definitions, and
  retained candidates.
- `CyclicalFractionalModel` with `fit`, `predict`, recursive prediction, and
  prediction intervals.

The runtime dependency footprint is intentionally small: the package depends on
NumPy only.

## Installation

From PyPI, once released:

```bash
python3 -m pip install cyclical-fractional-test
```

For local development:

```bash
git clone https://github.com/aslanda-design/log_memory_cycles.git
cd log_memory_cycles
python3 -m pip install -e ".[dev,docs]"
```

Python 3.11 or newer is required.

## Quickstart

```python
import numpy as np

from cyclical_fractional_test import (
    CyclicalTestConfig,
    compute_periodogram,
    run_cyclical_fractional_test,
)

rng = np.random.default_rng(42)
T = 240
t = np.arange(T, dtype=float)
y = np.cos(2.0 * np.pi * 12 * t / T) + 0.25 * rng.standard_normal(T)

lambdas, periodogram = compute_periodogram(y)

result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        n_deterministic_cycles=4,
        r_window=5,
        top_k=3,
        error_model="ar1",
    ),
)

best = result.best_result
print(best.cycles)
print(best.test_value)
print(result.diagnostics.n_candidates_evaluated)
```

By default, the test uses an adaptive `D` search. To evaluate a fixed Cartesian
grid instead:

```python
result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        r_window=5,
        top_k=3,
    ),
)
```

## Estimator API

`CyclicalFractionalModel` wraps the test in a scikit-learn-style interface.

```python
from cyclical_fractional_test import CyclicalFractionalModel

model = CyclicalFractionalModel(
    n_deterministic_cycles=4,
    error_model="ar1",
).fit(y)

in_sample = model.predict(len(y))
forecast = model.predict(len(y) + 20)
lower, upper = model.predict_interval(len(y) + 20, alpha=0.05)
```

The fitted model exposes selected-cycle attributes such as `cycles_`, `R_`,
`D_`, `betas_`, `ar_coefficients_`, `innovation_variance_`, and `result_`.

## Documentation

Markdown documentation lives in `docs/`:

- [Quickstart](docs/quickstart.md)
- [API reference](docs/api_reference.md)
- [Mathematical background](docs/mathematical_background.md)
- [Original test mapping](docs/original_test_mapping.md)
- [Data-flow diagram](docs/data_flow_diagram.md)
- [Implementation notes](docs/implementation_notes.md)
- [Development guide](docs/development.md)
- [Publishing guide](docs/publishing.md)

Preview the documentation locally with:

```bash
python3 -m mkdocs serve
```

## Development

Run tests:

```bash
python3 -m pytest
```

Run tests with coverage:

```bash
python3 -m pytest --cov=cyclical_fractional_test --cov-report=term-missing
```

Build and validate distribution artifacts:

```bash
python3 -m build
python3 -m twine check dist/*
```

Local datasets, notebooks, generated figures, and model artifacts are kept out
of the published package by `MANIFEST.in`.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
