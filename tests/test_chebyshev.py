import numpy as np
import pytest

from cyclical_fractional_test import InvalidConfigurationError
from cyclical_fractional_test.chebyshev import (
    build_chebyshev_design,
    build_chebyshev_design_at,
    build_single_chebyshev_polynomial,
    evaluate_single_chebyshev_polynomial,
)


# ---------------------------------------------------------------------------
# build_single_chebyshev_polynomial
# ---------------------------------------------------------------------------


def test_build_single_chebyshev_polynomial_order_zero_returns_ones():
    result = build_single_chebyshev_polynomial(T=5, order=0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (5,)
    np.testing.assert_allclose(result, np.ones(5))


def test_build_single_chebyshev_polynomial_order_one_matches_formula():
    T = 4
    t = np.arange(1, T + 1, dtype=float)
    expected = 2.0 * np.cos(1 * np.pi * (t - 0.5) / T)
    result = build_single_chebyshev_polynomial(T=T, order=1)
    np.testing.assert_allclose(result, expected)


def test_build_single_chebyshev_polynomial_order_two_matches_formula():
    T = 4
    t = np.arange(1, T + 1, dtype=float)
    expected = 2.0 * np.cos(2 * np.pi * (t - 0.5) / T)
    result = build_single_chebyshev_polynomial(T=T, order=2)
    np.testing.assert_allclose(result, expected)


def test_build_single_chebyshev_polynomial_larger_T():
    T = 100
    order = 3
    t = np.arange(1, T + 1, dtype=float)
    expected = 2.0 * np.cos(order * np.pi * (t - 0.5) / T)
    result = build_single_chebyshev_polynomial(T=T, order=order)
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# build_chebyshev_design — shape
# ---------------------------------------------------------------------------


def test_build_chebyshev_design_without_intercept_has_expected_shape():
    result = build_chebyshev_design(T=10, n_cycles=4, include_intercept=False)
    assert result.shape == (10, 4)


def test_build_chebyshev_design_with_intercept_has_expected_shape():
    result = build_chebyshev_design(T=10, n_cycles=4, include_intercept=True)
    assert result.shape == (10, 5)
    np.testing.assert_allclose(result[:, 0], np.ones(10))


def test_build_chebyshev_design_zero_cycles_without_intercept_has_no_columns():
    result = build_chebyshev_design(T=10, n_cycles=0, include_intercept=False)
    assert result.shape == (10, 0)


def test_build_chebyshev_design_zero_cycles_with_intercept_has_only_intercept():
    result = build_chebyshev_design(T=10, n_cycles=0, include_intercept=True)
    assert result.shape == (10, 1)
    np.testing.assert_allclose(result[:, 0], np.ones(10))


# ---------------------------------------------------------------------------
# build_chebyshev_design — column correctness
# ---------------------------------------------------------------------------


def test_build_chebyshev_design_columns_match_single_polynomials():
    T, n_cycles = 8, 3
    X = build_chebyshev_design(T=T, n_cycles=n_cycles, include_intercept=False)
    for col_idx, k in enumerate(range(1, n_cycles + 1)):
        expected = build_single_chebyshev_polynomial(T=T, order=k)
        np.testing.assert_allclose(X[:, col_idx], expected)


def test_build_chebyshev_design_with_intercept_columns_match():
    T, n_cycles = 8, 3
    X = build_chebyshev_design(T=T, n_cycles=n_cycles, include_intercept=True)
    np.testing.assert_allclose(X[:, 0], np.ones(T))
    for col_idx, k in enumerate(range(1, n_cycles + 1), start=1):
        expected = build_single_chebyshev_polynomial(T=T, order=k)
        np.testing.assert_allclose(X[:, col_idx], expected)


def test_build_chebyshev_design_n_cycles_1_no_intercept():
    result = build_chebyshev_design(T=5, n_cycles=1, include_intercept=False)
    assert result.shape == (5, 1)
    np.testing.assert_allclose(
        result[:, 0], build_single_chebyshev_polynomial(T=5, order=1)
    )


def test_build_chebyshev_design_uses_explicit_orders_without_intercept():
    X = build_chebyshev_design(
        T=8,
        n_cycles=99,
        include_intercept=False,
        orders=(2, 5),
    )

    assert X.shape == (8, 2)
    np.testing.assert_allclose(X[:, 0], build_single_chebyshev_polynomial(8, 2))
    np.testing.assert_allclose(X[:, 1], build_single_chebyshev_polynomial(8, 5))


def test_build_chebyshev_design_uses_explicit_orders_with_intercept():
    X = build_chebyshev_design(
        T=8,
        n_cycles=99,
        include_intercept=True,
        orders=(2, 5),
    )

    assert X.shape == (8, 3)
    np.testing.assert_allclose(X[:, 0], np.ones(8))
    np.testing.assert_allclose(X[:, 1], build_single_chebyshev_polynomial(8, 2))
    np.testing.assert_allclose(X[:, 2], build_single_chebyshev_polynomial(8, 5))


# ---------------------------------------------------------------------------
# Error cases — build_single_chebyshev_polynomial
# ---------------------------------------------------------------------------


def test_build_single_chebyshev_polynomial_rejects_zero_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=0, order=1)


def test_build_single_chebyshev_polynomial_rejects_negative_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=-3, order=1)


def test_build_single_chebyshev_polynomial_rejects_float_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=5.0, order=1)  # type: ignore


def test_build_single_chebyshev_polynomial_rejects_bool_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=True, order=1)  # type: ignore


def test_build_single_chebyshev_polynomial_rejects_negative_order():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=5, order=-1)


def test_build_single_chebyshev_polynomial_rejects_float_order():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=5, order=1.5)  # type: ignore


# ---------------------------------------------------------------------------
# Error cases — build_chebyshev_design
# ---------------------------------------------------------------------------


def test_build_chebyshev_design_rejects_non_positive_T():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=-1, n_cycles=4)


def test_build_chebyshev_design_rejects_float_T():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10.0, n_cycles=4)  # type: ignore


def test_build_chebyshev_design_rejects_negative_n_cycles():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=-2)


def test_build_chebyshev_design_rejects_float_n_cycles():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=2.5)  # type: ignore


def test_build_chebyshev_design_rejects_non_boolean_intercept():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=4, include_intercept=1)  # type: ignore


def test_build_chebyshev_design_rejects_string_intercept():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=4, include_intercept="yes")  # type: ignore


@pytest.mark.parametrize("bad_orders", [(), (0,), (-1,), (2, 2), (True,), (1.5,)])
def test_build_chebyshev_design_rejects_bad_explicit_orders(bad_orders):
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=4, orders=bad_orders)


# ---------------------------------------------------------------------------
# build_chebyshev_design_at / evaluate_single_chebyshev_polynomial
# ---------------------------------------------------------------------------


def test_design_at_matches_in_sample_grid():
    T = 30
    t = np.arange(1, T + 1, dtype=float)
    expected = build_chebyshev_design(T, 4, include_intercept=True)
    got = build_chebyshev_design_at(t, T, 4, include_intercept=True)
    np.testing.assert_allclose(got, expected)


def test_design_at_matches_explicit_order_in_sample_grid():
    T = 30
    t = np.arange(1, T + 1, dtype=float)
    expected = build_chebyshev_design(T, 4, include_intercept=True, orders=(3, 9))
    got = build_chebyshev_design_at(t, T, 4, include_intercept=True, orders=(3, 9))
    np.testing.assert_allclose(got, expected)


def test_design_at_zero_cycles_matches_in_sample_grid():
    T = 30
    t = np.arange(1, T + 1, dtype=float)
    expected = build_chebyshev_design(T, 0, include_intercept=True)
    got = build_chebyshev_design_at(t, T, 0, include_intercept=True)
    np.testing.assert_allclose(got, expected)


def test_evaluate_polynomial_matches_in_sample():
    T = 25
    t = np.arange(1, T + 1, dtype=float)
    for k in (0, 1, 3):
        np.testing.assert_allclose(
            evaluate_single_chebyshev_polynomial(t, T, k),
            build_single_chebyshev_polynomial(T, k),
        )


def test_design_at_extrapolates_beyond_T():
    T = 20
    t_future = np.arange(T + 1, T + 6, dtype=float)
    X_future = build_chebyshev_design_at(t_future, T, 3, include_intercept=False)
    assert X_future.shape == (5, 3)
    assert np.all(np.isfinite(X_future))


def test_design_at_rejects_empty_t():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design_at(np.array([]), 20, 3, include_intercept=False)
