"""
Legacy utilities module.

Contains older utility functions that predate the geometric algebra framework.
These are kept for backward compatibility and use in specific contexts
(e.g., visualization transforms).
"""

# Vector utilities
# Coordinate utilities
from morphis._legacy.coordinates import (
    coordinate_grid,
    to_cartesian,
    to_spherical,
)

# Rotation utilities
from morphis._legacy.rotations import (
    E1,
    E2,
    E3,
    STANDARD_FRAME,
    apply_rotation,
    euler_angles_zyx,
    extrinsic_rotation,
    intrinsic_rotation,
    reset_blade_transform,
    rotate,
    rotate_blade,
    rotate_frame,
    rotation_matrix,
    set_blade_position,
    solve_rotation_angle,
    translate_blade,
)

# Smoothing utilities
from morphis._legacy.smoothing import (
    SMOOTHERS,
    Smoother,
    get_smoother,
    smooth_in_out_cubic,
    smooth_in_out_quad,
    smooth_in_out_sine,
    smooth_in_quad,
    smooth_linear,
    smooth_out_quad,
)
from morphis._legacy.vectors import (
    cross,
    dot,
    kronecker_delta,
    levi_civita,
    mag,
    project_onto_axis,
    unit,
)


__all__ = [
    # Vectors
    "cross",
    "dot",
    "kronecker_delta",
    "levi_civita",
    "mag",
    "project_onto_axis",
    "unit",
    # Rotations
    "E1",
    "E2",
    "E3",
    "STANDARD_FRAME",
    "apply_rotation",
    "euler_angles_zyx",
    "extrinsic_rotation",
    "intrinsic_rotation",
    "reset_blade_transform",
    "rotate",
    "rotate_blade",
    "rotate_frame",
    "rotation_matrix",
    "set_blade_position",
    "solve_rotation_angle",
    "translate_blade",
    # Coordinates
    "coordinate_grid",
    "to_cartesian",
    "to_spherical",
    # Smoothing
    "SMOOTHERS",
    "Smoother",
    "get_smoother",
    "smooth_in_out_cubic",
    "smooth_in_out_quad",
    "smooth_in_out_sine",
    "smooth_in_quad",
    "smooth_linear",
    "smooth_out_quad",
]
