class CyclicalFractionalTestError(Exception):
    """Base class for package-specific errors."""


class InvalidSeriesError(CyclicalFractionalTestError):
    """The input series Y(t) is not usable by the test."""


class InvalidConfigurationError(CyclicalFractionalTestError):
    """A configuration value is outside the supported range."""


class InvalidCycleError(CyclicalFractionalTestError):
    """A stochastic cycle, or a group of cycles, is inconsistent."""


class NotFittedError(CyclicalFractionalTestError):
    """A model method requiring a prior fit was called before fitting."""
