"""
Geometric Algebra - Transforms

Transformation constructors and actions: rotors, translators, motors,
and convenience functions for applying transformations.
"""

# Rotations
# Actions
from morphis.transforms.actions import (
    rotate as rotate,
    transform as transform,
    translate as translate,
)

# Projective (PGA) operations
from morphis.transforms.projective import (
    are_collinear as are_collinear,
    are_coplanar as are_coplanar,
    bulk as bulk,
    direction as direction,
    distance_point_to_line as distance_point_to_line,
    distance_point_to_plane as distance_point_to_plane,
    distance_point_to_point as distance_point_to_point,
    euclidean as euclidean,
    is_direction as is_direction,
    is_point as is_point,
    line as line,
    line_in_plane as line_in_plane,
    plane as plane,
    plane_from_point_and_line as plane_from_point_and_line,
    point as point,
    point_on_line as point_on_line,
    point_on_plane as point_on_plane,
    screw as screw,
    translator as translator,
    weight as weight,
)
from morphis.transforms.rotations import (
    rotation_about_point as rotation_about_point,
    rotor as rotor,
)
