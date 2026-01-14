"""
Geometric Algebra - Transformations

Constructor functions for rigid transformations (rotors, translators)
and action functions that apply them via sandwich products.

All transformations use the sandwich product: x' = M x ~M
"""

from morphis.geometry.transforms.actions import rotate, transform, translate
from morphis.geometry.transforms.constructors import (
    rotation_about_point,
    rotor,
    screw_motion,
    translator,
)


__all__ = [
    # Constructors
    "rotor",
    "translator",
    "rotation_about_point",
    "screw_motion",
    # Actions
    "rotate",
    "translate",
    "transform",
]
