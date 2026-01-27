"""
Geometric Algebra - Transformation Actions

Action functions that construct and apply transformations in one step.
These are convenience wrappers around the constructor functions.

All transformations use the sandwich product: x' = M x ~M
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from morphis.elements.metric import Metric
from morphis.operations.products import geometric, grade_project, reverse
from morphis.transforms.projective import translator
from morphis.transforms.rotations import rotor


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Rotate Action
# =============================================================================


def rotate(
    b: Vector,
    B: Vector,
    angle: float | NDArray,
) -> Vector:
    """
    Rotate a vector by angle in the plane defined by bivector B.

    Creates a rotor and applies it via sandwich product: M * b * ~M

    Args:
        b: Vector to rotate (any grade).
        B: Bivector defining the rotation plane.
        angle: Rotation angle in radians.

    Returns:
        Rotated vector of same grade.

    Example:
        v_rotated = rotate(v, e1 ^ e2, pi/4)
    """
    # Validate compatible metrics
    Metric.merge(b.metric, B.metric)

    M = rotor(B, angle)

    # Sandwich product: M * b * ~M
    M_rev = reverse(M)
    temp = geometric(M, b)
    result = geometric(temp, M_rev)

    return grade_project(result, b.grade)


# =============================================================================
# Translate Action
# =============================================================================


def translate(u: Vector, v: Vector) -> Vector:
    """
    Translate a vector by a direction (PGA only).

    Creates a translator and applies it via sandwich product: M * u * ~M

    Args:
        u: PGA vector to translate (any grade).
        v: Direction Vector representing the translation displacement.

    Returns:
        Translated vector of same grade.

    Example:
        from morphis.transforms.projective import direction
        d = direction([1, 0, 0])
        p_translated = translate(p, d)
    """
    Metric.merge(u.metric, v.metric)
    M = translator(v)

    # Sandwich product: M * u * ~M
    M_rev = reverse(M)
    temp = geometric(M, u)
    result = geometric(temp, M_rev)

    return grade_project(result, u.grade)


# =============================================================================
# Transform Action
# =============================================================================


def transform(
    b: Vector,
    M: MultiVector,
) -> Vector:
    """
    Apply a motor/versor transformation to a vector via sandwich product.

    Computes: M * b * ~M

    Args:
        b: Vector to transform (any grade).
        M: Motor (MultiVector with grades {0, 2}) representing the transformation.

    Returns:
        Transformed vector of same grade.

    Example:
        M = rotor(B, pi/2)
        v_transformed = transform(v, M)
    """
    # Validate compatible metrics
    Metric.merge(b.metric, M.metric)

    # Sandwich product: M * b * ~M
    M_rev = reverse(M)
    temp = geometric(M, b)
    result = geometric(temp, M_rev)

    return grade_project(result, b.grade)
