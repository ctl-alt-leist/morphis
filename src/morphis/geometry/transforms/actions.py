"""
Geometric Algebra - Transformation Actions

Action functions that construct and apply transformations in one step.
These are convenience wrappers around the constructor functions.

All transformations use the sandwich product: x' = M x ~M
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from morphis.geometry.algebra.geometric import geometric, grade_project, reverse
from morphis.geometry.model.metric import Metric
from morphis.geometry.transforms.constructors import rotor, translator


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.multivector import MultiVector


# =============================================================================
# Rotate Action
# =============================================================================


def rotate(
    b: Blade,
    B: Blade,
    angle: float | NDArray,
) -> Blade:
    """
    Rotate a blade by angle in the plane defined by bivector B.

    Creates a rotor and applies it via sandwich product: M @ b @ ~M

    Args:
        b: Blade to rotate (any grade).
        B: Bivector defining the rotation plane.
        angle: Rotation angle in radians.

    Returns:
        Rotated blade of same grade.

    Example:
        v_rotated = rotate(v, e1 ^ e2, pi/4)
    """
    # Validate compatible metrics
    Metric.merge(b.metric, B.metric)

    M = rotor(B, angle)

    # Sandwich product: M @ b @ ~M
    M_rev = reverse(M)
    temp = geometric(M, b)
    result = geometric(temp, M_rev)

    return grade_project(result, b.grade)


# =============================================================================
# Translate Action
# =============================================================================


def translate(
    b: Blade,
    displacement: NDArray,
) -> Blade:
    """
    Translate a blade by displacement vector (PGA only).

    Creates a translator and applies it via sandwich product: M @ b @ ~M

    Args:
        b: PGA blade to translate (any grade).
        displacement: Translation vector.

    Returns:
        Translated blade of same grade.

    Example:
        p_translated = translate(p, [1, 0, 0])
    """
    M = translator(displacement, metric=b.metric)

    # Sandwich product: M @ b @ ~M
    M_rev = reverse(M)
    temp = geometric(M, b)
    result = geometric(temp, M_rev)

    return grade_project(result, b.grade)


# =============================================================================
# Transform Action
# =============================================================================


def transform(
    b: Blade,
    M: MultiVector,
) -> Blade:
    """
    Apply a motor/versor transformation to a blade via sandwich product.

    Computes: M @ b @ ~M

    Args:
        b: Blade to transform (any grade).
        M: Motor (MultiVector with grades {0, 2}) representing the transformation.

    Returns:
        Transformed blade of same grade.

    Example:
        M = rotor(B, pi/2)
        v_transformed = transform(v, M)
    """
    # Validate compatible metrics
    Metric.merge(b.metric, M.metric)

    # Sandwich product: M @ b @ ~M
    M_rev = reverse(M)
    temp = geometric(M, b)
    result = geometric(temp, M_rev)

    return grade_project(result, b.grade)
