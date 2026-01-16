"""
Geometric Algebra - Rotation Constructors

Constructor functions for rotors that return MultiVector objects. These work
with the geometric product operator `@` for transformations via sandwich
products: rotated = M @ b @ ~M

All operations support collection dimensions via einsum broadcasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array, broadcast_shapes, cos, newaxis, sin

from morphis.operations.products import geometric


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from morphis.elements.blade import Blade
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Rotor Constructor
# =============================================================================


def rotor(B: Blade, angle: float | NDArray) -> MultiVector:
    """
    Create a rotor for pure rotation about the origin.

    M = exp(-B theta/2) = cos(theta/2) - sin(theta/2) B

    The rotor is a MultiVector with grades {0, 2}. Apply via sandwich product:
        rotated = M @ b @ ~M

    Args:
        B: Bivector (grade-2) defining the rotation plane. Should be normalized
           (unit bivector) for angle to be exact rotation angle.
        angle: Rotation angle in radians. Supports scalar or array for batch.

    Returns:
        MultiVector with grades {0, 2} representing the rotation.

    Example:
        # Create rotor for 90-degree rotation in xy-plane
        B = e1 ^ e2  # bivector
        M = rotor(B, pi/2)
        v_rotated = M @ v @ ~M
    """
    from morphis.elements.blade import Blade
    from morphis.elements.multivector import MultiVector

    angle = array(angle)
    metric = B.metric

    # Determine collection shape
    if angle.ndim == 0:
        collection = B.collection
        half_angle = angle / 2
    else:
        collection = angle.shape
        half_angle = angle / 2

    # Scalar part: cos(theta/2)
    scalar_data = cos(half_angle)

    # Bivector part: -sin(theta/2) B
    if angle.ndim == 0:
        bivector_data = -sin(half_angle) * B.data
    else:
        sin_expanded = sin(half_angle)
        for _ in range(B.grade):
            sin_expanded = sin_expanded[..., newaxis]
        bivector_data = -sin_expanded * B.data

    components = {
        0: Blade(scalar_data, grade=0, metric=metric, collection=collection),
        2: Blade(data=bivector_data, grade=2, metric=metric, collection=collection),
    }

    return MultiVector(data=components, metric=metric, collection=collection)


# =============================================================================
# Rotation About Point Constructor
# =============================================================================


def rotation_about_point(
    p: Blade,
    B: Blade,
    angle: float | NDArray,
) -> MultiVector:
    """
    Create a motor for rotation about an arbitrary center point (PGA).

    Implemented as composition: translate to origin, rotate, translate back.
    M = T2 @ R @ T1 where T1 = translator(-c), R = rotor(B, theta), T2 = translator(c)

    Args:
        p: PGA point (grade-1) defining the rotation center.
        B: Bivector defining the rotation plane.
        angle: Rotation angle in radians.

    Returns:
        MultiVector (motor) representing rotation about the center.

    Example:
        p = point([1, 0, 0])  # Center at x=1
        B = e1 ^ e2
        M = rotation_about_point(p, B, pi/2)
        v_rotated = M @ v @ ~M
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector
    from morphis.transforms.projective import euclidean as to_euclidean, translator

    # Validate metrics
    metric = Metric.merge(p.metric, B.metric)

    # Extract center coordinates
    c = to_euclidean(p)  # Shape: (..., d)

    # Create three components
    T1 = translator(-c, metric=metric)  # Translate to origin
    R = rotor(B, angle)  # Rotate
    T2 = translator(c, metric=metric)  # Translate back

    # Compose via geometric product: T2 @ R @ T1
    temp = geometric(R, T1)
    result = geometric(T2, temp)

    # Project to motor grades {0, 2}
    motor_components = {k: v for k, v in result.data.items() if k in {0, 2}}

    return MultiVector(
        components=motor_components,
        metric=metric,
        collection=broadcast_shapes(p.collection, B.collection),
    )
