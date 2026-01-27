"""
Geometric Algebra - Utilities

Utility functions for vector validation, dimension checking, and broadcasting.

Vector naming convention: u, v, w (never a, b, c for vectors).
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

from numpy import broadcast_shapes


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


F = TypeVar("F", bound=Callable)


# =============================================================================
# Vector Dimension Helpers
# =============================================================================


def get_common_dim(*vectors: Vector) -> int:
    """
    Get the common dimension d of a collection of vectors.

    Raises ValueError if vectors have different dimensions or if no vectors
    are provided.

    Returns the common dimension d.
    """
    if not vectors:
        raise ValueError("At least one vector required")

    dims = [u.dim for u in vectors]
    d = dims[0]

    if not all(dim == d for dim in dims):
        raise ValueError(f"Dimension mismatch: vectors have dimensions {dims}")

    return d


def get_broadcast_collection(*vectors: Vector) -> tuple[int, ...]:
    """
    Compute the broadcast-compatible collection shape for multiple vectors.

    Uses numpy-style broadcasting rules to determine the result collection shape.

    Returns the broadcasted collection shape.
    """
    if not vectors:
        raise ValueError("At least one vector required")

    return broadcast_shapes(*(u.collection for u in vectors))


def validate_same_dim(*vectors: Vector) -> None:
    """
    Validate that all vectors have the same dimension d.

    Raises ValueError if dimensions differ.
    """
    if len(vectors) < 2:
        return

    d = vectors[0].dim
    for k, u in enumerate(vectors[1:], start=1):
        if u.dim != d:
            raise ValueError(f"Dimension mismatch: vector 0 has dim {d}, vector {k} has dim {u.dim}")


def same_dim(func: F) -> F:
    """
    Decorator that validates all Vector arguments have the same dimension.

    Applies to functions where positional arguments are Vectors. Validates
    dimensions before calling the function.

    Example:
        @same_dim
        def wedge(*vectors: Vector) -> Vector:
            ...
    """
    from morphis.elements.vector import Vector

    @wraps(func)
    def wrapper(*args, **kwargs):
        vectors = [arg for arg in args if isinstance(arg, Vector)]
        validate_same_dim(*vectors)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# =============================================================================
# Collection Shape Helpers
# =============================================================================


def broadcast_collection_shape(*vectors: Vector) -> tuple[int, ...]:
    """
    Compute the broadcast shape of collection dimensions for multiple vectors.

    Uses numpy-style broadcasting rules: dimensions must be equal or one of
    them must be 1.

    Returns the broadcast collection shape.

    Note: This is an alias for get_broadcast_collection for backwards compatibility.
    """
    return get_broadcast_collection(*vectors)
