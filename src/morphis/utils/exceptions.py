"""Exception utilities for morphis."""

import warnings
from functools import wraps
from typing import Callable

from numpy import iscomplexobj


class ComplexUnsupportedError(AssertionError):
    """Error raised when receiving unsupported complex dtypes."""

    def __init__(self, message, error):
        super().__init__(message)
        self.error = error


def complex_error_message(**kwargs):
    """
    Returns the unsupported complex input string for all objects given in the
    kwargs which are complex.
    """
    complex_inputs = {k: v for k, v in kwargs.items() if iscomplexobj(v)}
    return f"Unsupported complex inputs: {complex_inputs}"


def suppress_runtime_warning(func: Callable) -> Callable:
    """Decorator to suppress runtime warnings within a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(*args, **kwargs)
        return result

    return wrapper
