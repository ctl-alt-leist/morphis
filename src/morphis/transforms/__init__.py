"""
Geometric Algebra - Transforms

Transformation constructors and actions: rotors, translators, motors,
and convenience functions for applying transformations.
"""

# Rotations
# Actions
from morphis.transforms.actions import (
    rotate,
    transform,
    translate,
)

# Projective (PGA) operations
from morphis.transforms.projective import (
    are_collinear,
    are_coplanar,
    bulk,
    direction,
    distance_point_to_line,
    distance_point_to_plane,
    distance_point_to_point,
    euclidean,
    is_direction,
    is_point,
    line,
    line_in_plane,
    plane,
    plane_from_point_and_line,
    point,
    point_on_line,
    point_on_plane,
    screw_motion,
    translator,
    weight,
)
from morphis.transforms.rotations import (
    rotation_about_point,
    rotor,
)


__all__ = [
    # Rotations
    "rotor",
    "rotation_about_point",
    # Projective
    "point",
    "direction",
    "weight",
    "bulk",
    "euclidean",
    "is_point",
    "is_direction",
    "line",
    "plane",
    "plane_from_point_and_line",
    "distance_point_to_point",
    "distance_point_to_line",
    "distance_point_to_plane",
    "are_collinear",
    "are_coplanar",
    "point_on_line",
    "point_on_plane",
    "line_in_plane",
    "translator",
    "screw_motion",
    # Actions
    "rotate",
    "translate",
    "transform",
]
