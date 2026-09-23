class DataValidationError(ValueError):
    """Input data cannot be used for portfolio calculations."""


class InsufficientDataError(DataValidationError):
    """Too few common observations remain after alignment."""


class UndefinedMetricError(ValueError):
    """The requested metric has no finite, defined value."""
