"""
Legacy utilities module.

Contains older utility functions that predate the geometric algebra framework.
These are kept for backward compatibility and use in specific contexts
(e.g., visualization transforms).
"""

# Vector utilities
# Coordinate utilities
from morphis._legacy.coordinates import (
    coordinate_grid as coordinate_grid,
    to_cartesian as to_cartesian,
    to_spherical as to_spherical,
)

# Rotation utilities
from morphis._legacy.rotations import (
    E1 as E1,
    E2 as E2,
    E3 as E3,
    STANDARD_FRAME as STANDARD_FRAME,
    apply_rotation as apply_rotation,
    euler_angles_zyx as euler_angles_zyx,
    extrinsic_rotation as extrinsic_rotation,
    intrinsic_rotation as intrinsic_rotation,
    reset_blade_transform as reset_blade_transform,
    rotate as rotate,
    rotate_blade as rotate_blade,
    rotate_frame as rotate_frame,
    rotation_matrix as rotation_matrix,
    set_blade_position as set_blade_position,
    solve_rotation_angle as solve_rotation_angle,
    translate_blade as translate_blade,
)

# Smoothing utilities
from morphis._legacy.smoothing import (
    SMOOTHERS as SMOOTHERS,
    Smoother as Smoother,
    get_smoother as get_smoother,
    smooth_in_out_cubic as smooth_in_out_cubic,
    smooth_in_out_quad as smooth_in_out_quad,
    smooth_in_out_sine as smooth_in_out_sine,
    smooth_in_quad as smooth_in_quad,
    smooth_linear as smooth_linear,
    smooth_out_quad as smooth_out_quad,
)
from morphis._legacy.vectors import (
    cross as cross,
    dot as dot,
    kronecker_delta as kronecker_delta,
    levi_civita as levi_civita,
    mag as mag,
    project_onto_axis as project_onto_axis,
    unit as unit,
)
