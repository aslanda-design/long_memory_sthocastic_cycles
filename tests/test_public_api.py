from importlib import metadata, resources

import cyclical_fractional_test as cft


def test_declared_public_exports_resolve_and_are_unique():
    assert len(cft.__all__) == len(set(cft.__all__))
    assert all(hasattr(cft, name) for name in cft.__all__)


def test_distribution_version_matches_imported_version():
    assert cft.__version__ == metadata.version("cyclical-fractional-test")


def test_package_includes_py_typed_marker():
    marker = resources.files("cyclical_fractional_test").joinpath("py.typed")
    assert marker.is_file()


def test_public_exception_hierarchy_is_stable():
    assert issubclass(cft.InvalidSeriesError, cft.CyclicalFractionalTestError)
    assert issubclass(cft.InvalidConfigurationError, cft.CyclicalFractionalTestError)
    assert issubclass(cft.InvalidCycleError, cft.CyclicalFractionalTestError)
    assert issubclass(cft.NotFittedError, cft.CyclicalFractionalTestError)


def test_public_surface_contains_main_user_entry_points():
    expected = {
        "CyclicalTestConfig",
        "CyclicalFractionalModel",
        "run_cyclical_fractional_test",
        "compute_periodogram",
        "compute_autocorrelogram",
        "StochasticCycle",
        "GridCandidateResult",
        "CyclicalFractionalTestResult",
    }

    assert expected.issubset(set(cft.__all__))
    assert all(getattr(cft, name) is not None for name in expected)
