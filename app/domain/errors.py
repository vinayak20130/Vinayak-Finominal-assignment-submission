class DataValidationError(ValueError):
    """Input data cannot be used for portfolio calculations."""


class InsufficientDataError(DataValidationError):
    """Too few common observations remain after alignment."""


class UndefinedMetricError(ValueError):
    """The requested metric has no finite, defined value."""


class InfeasibleConstraintsError(ValueError):
    """The requested constraints provably admit no valid allocation."""


class EqualWeightConflictError(ValueError):
    """Equal weights violate a supplied bound or portfolio constraint."""


class OptimizationFailedError(RuntimeError):
    """No verified feasible solution was found; infeasibility is not proven."""


class TickerNotFoundError(LookupError):
    """A requested ticker is not in the market data."""


class DataUnavailableError(RuntimeError):
    """The market data store cannot be reached."""
