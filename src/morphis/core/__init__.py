"""
Core mathematical utilities for vector operations, coordinates, and rotations.

This module provides fundamental operations on vectors and rotation matrices,
with support for arbitrary dimensions where mathematically possible.
"""

from morphis.core.coordinates import coordinate_grid, to_cartesian, to_spherical
from morphis.core.smoothing import (
    Smoother,
    get_smoother,
    smooth_in_out_cubic,
    smooth_in_out_quad,
    smooth_in_out_sine,
    smooth_in_quad,
    smooth_linear,
    smooth_out_quad,
)
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
    # Coordinates
    "coordinate_grid",
    "to_cartesian",
    "to_spherical",
    # Smoothing
    "get_smoother",
    "smooth_in_out_cubic",
    "smooth_in_out_quad",
    "smooth_in_out_sine",
    "smooth_in_quad",
    "smooth_linear",
    "smooth_out_quad",
    "Smoother",
    # Vectors
    "cross",
    "dot",
    "kronecker_delta",
    "levi_civita",
    "mag",
    "project_onto_axis",
    "unit",
]
