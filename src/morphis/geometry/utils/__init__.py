"""
Geometric Algebra - Utilities

Utility functions for blade validation, dimension checking, and broadcasting.
"""

from morphis.geometry.utils.helpers import (
    broadcast_collection_shape,
    get_broadcast_collection,
    get_common_dim,
    same_dim,
    validate_same_dim,
)


__all__ = [
    "broadcast_collection_shape",
    "get_broadcast_collection",
    "get_common_dim",
    "same_dim",
    "validate_same_dim",
]
