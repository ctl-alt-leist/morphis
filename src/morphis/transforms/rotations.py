"""
Geometric Algebra - Rotation Constructors

Constructor functions for rotors that return MultiVector objects. These work
with the geometric product operator `*` for transformations via sandwich
products: rotated = M * b * ~M

All operations support collection dimensions via einsum broadcasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array, broadcast_shapes, newaxis

from morphis.operations.products import geometric


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Rotor Constructor
# =============================================================================


def rotor(B: Vector, angle: float | NDArray) -> MultiVector:
    """
    Create a rotor for pure rotation about the origin.

    M = exp(-B * angle/2)

    Implemented via exp_vector(), which provides closed-form evaluation
    for any metric signature (Euclidean, Lorentzian, or degenerate).

    The rotor is a MultiVector with grades {0, 2}. Apply via sandwich product:
        rotated = M * b * ~M

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
        v_rotated = M * v * ~M
    """
    from morphis.elements.vector import Vector
    from morphis.operations.exponential import exp_vector

    angle = array(angle)

    # Compute generator: -B * angle/2
    # Handle array angles by proper broadcasting
    if angle.ndim == 0:
        # Scalar angle
        generator = B * (-float(angle) / 2)
    else:
        # Array of angles: need to expand for broadcasting
        half_angle = -angle / 2
        for _ in range(B.grade):
            half_angle = half_angle[..., newaxis]

        generator = Vector(
            half_angle * B.data,
            grade=B.grade,
            metric=B.metric,
            collection=angle.shape + B.collection,
        )

    return exp_vector(generator)


# =============================================================================
# Rotation About Point Constructor
# =============================================================================


def rotation_about_point(
    p: Vector,
    B: Vector,
    angle: float | NDArray,
) -> MultiVector:
    """
    Create a motor for rotation about an arbitrary center point (PGA).

    Implemented as composition: translate to origin, rotate, translate back.
    M = T2 * R * T1 where T1 = translator(-c), R = rotor(B, theta), T2 = translator(c)

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
        v_rotated = M * v * ~M
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector
    from morphis.transforms.projective import direction, euclidean as to_euclidean, translator

    # Validate metrics
    metric = Metric.merge(p.metric, B.metric)

    # Extract center coordinates and create direction vectors
    c = to_euclidean(p)  # Shape: (..., d)
    neg_c = direction(-c, metric=metric, collection=p.collection)
    pos_c = direction(c, metric=metric, collection=p.collection)

    # Create three components
    T1 = translator(neg_c)  # Translate to origin
    R = rotor(B, angle)  # Rotate
    T2 = translator(pos_c)  # Translate back

    # Compose via geometric product: T2 * R * T1
    temp = geometric(R, T1)
    result = geometric(T2, temp)

    # Project to motor grades {0, 2}
    motor_components = {k: v for k, v in result.data.items() if k in {0, 2}}

    return MultiVector(
        data=motor_components,
        metric=metric,
        collection=broadcast_shapes(p.collection, B.collection),
    )
