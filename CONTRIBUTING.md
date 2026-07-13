# Contributing

Thanks for improving `cyclical-fractional-test`.

## Development Setup

```bash
python3 -m pip install -e ".[dev,docs]"
```

## Checks

Run the test suite before opening a pull request:

```bash
python3 -m pytest
```

Build and validate the distribution artifacts when packaging metadata changes:

```bash
python3 -m build
python3 -m twine check dist/*
```

Preview the documentation locally:

```bash
python3 -m mkdocs serve
```

## Code Style

- Keep runtime dependencies small. The core package currently depends only on
  NumPy.
- Keep public behavior covered by focused tests under `tests/`.
- Prefer explicit validation errors from `cyclical_fractional_test.exceptions`.
- Keep notebooks, generated figures, model artifacts, and local datasets out of
  published distributions.
