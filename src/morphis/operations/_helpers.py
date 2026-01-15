"""
Geometric Algebra - Utilities

Utility functions for blade validation, dimension checking, and broadcasting.

Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

from numpy import broadcast_shapes


if TYPE_CHECKING:
    from morphis.elements.blade import Blade


F = TypeVar("F", bound=Callable)


# =============================================================================
# Blade Dimension Helpers
# =============================================================================


def get_common_dim(*blades: Blade) -> int:
    """
    Get the common dimension d of a collection of blades.

    Raises ValueError if blades have different dimensions or if no blades
    are provided.

    Returns the common dimension d.
    """
    if not blades:
        raise ValueError("At least one blade required")

    dims = [u.dim for u in blades]
    d = dims[0]

    if not all(dim == d for dim in dims):
        raise ValueError(f"Dimension mismatch: blades have dimensions {dims}")

    return d


def get_broadcast_collection(*blades: Blade) -> tuple[int, ...]:
    """
    Compute the broadcast-compatible collection shape for multiple blades.

    Uses numpy-style broadcasting rules to determine the result collection shape.

    Returns the broadcasted collection shape.
    """
    if not blades:
        raise ValueError("At least one blade required")

    return broadcast_shapes(*(u.collection for u in blades))


def validate_same_dim(*blades: Blade) -> None:
    """
    Validate that all blades have the same dimension d.

    Raises ValueError if dimensions differ.
    """
    if len(blades) < 2:
        return

    d = blades[0].dim
    for k, u in enumerate(blades[1:], start=1):
        if u.dim != d:
            raise ValueError(f"Dimension mismatch: blade 0 has dim {d}, blade {k} has dim {u.dim}")


def same_dim(func: F) -> F:
    """
    Decorator that validates all Blade arguments have the same dimension.

    Applies to functions where positional arguments are Blades. Validates
    dimensions before calling the function.

    Example:
        @same_dim
        def wedge(*blades: Blade) -> Blade:
            ...
    """
    from morphis.elements.blade import Blade

    @wraps(func)
    def wrapper(*args, **kwargs):
        blades = [arg for arg in args if isinstance(arg, Blade)]
        validate_same_dim(*blades)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# =============================================================================
# Collection Shape Helpers
# =============================================================================


def broadcast_collection_shape(*blades: Blade) -> tuple[int, ...]:
    """
    Compute the broadcast shape of collection dimensions for multiple blades.

    Uses numpy-style broadcasting rules: dimensions must be equal or one of
    them must be 1.

    Returns the broadcast collection shape.

    Note: This is an alias for get_broadcast_collection for backwards compatibility.
    """
    return get_broadcast_collection(*blades)
