"""
Core mathematical utilities for vector operations, coordinates, and rotations.

This module provides fundamental operations on vectors and rotation matrices,
with support for arbitrary dimensions where mathematically possible.
"""

from morphis.core.coordinates import coordinate_grid, to_cartesian, to_spherical
from morphis.core.vectors import (
    cross,
    dot,
    kronecker_delta,
    levi_civita,
    mag,
    project_onto_axis,
    unit,
)


__all__ = [
    "coordinate_grid",
    "cross",
    "dot",
    "kronecker_delta",
    "levi_civita",
    "mag",
    "project_onto_axis",
    "to_cartesian",
    "to_spherical",
    "unit",
]
