"""
Geometric Algebra - Transformation Constructors

Constructor functions for rigid transformations (rotors, translators, motors)
that return MultiVector objects. These work with the geometric product operator
`@` for transformations via sandwich products: rotated = M @ b @ ~M

All operations support collection dimensions via einsum broadcasting.
Stay within the GA mindset - use pure geometric algebra operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array, cos, newaxis, ones, sin, zeros
from numpy.typing import NDArray

from morphis.geometry.model.metric import Metric, pga


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.multivector import MultiVector


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
    from morphis.geometry.model.blade import Blade, scalar_blade
    from morphis.geometry.model.multivector import MultiVector

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
        0: scalar_blade(scalar_data, metric=metric, collection=collection),
        2: Blade(data=bivector_data, grade=2, metric=metric, collection=collection),
    }

    return MultiVector(components=components, metric=metric, collection=collection)


# =============================================================================
# Translator Constructor
# =============================================================================


def translator(
    displacement: NDArray, metric: Metric | None = None, collection: tuple[int, ...] | None = None
) -> MultiVector:
    """
    Create a translator for pure translation (PGA only).

    M = 1 + (1/2) t^m e_{0m}

    The translator is a MultiVector with grades {0, 2} where the bivector
    part uses only degenerate (e_0) components. Apply via sandwich product:
        translated = M @ p @ ~M

    Args:
        displacement: Translation vector of shape (..., d) where d is the
                     Euclidean dimension.
        metric: PGA metric. Inferred from displacement if not provided.
        collection: Shape of the collection dimensions.

    Returns:
        MultiVector with grades {0, 2} representing the translation.

    Example:
        # Translate by (1, 0, 0) in 3D PGA
        M = translator([1, 0, 0])
        p_translated = M @ p @ ~M
    """
    from morphis.geometry.model.blade import Blade, scalar_blade
    from morphis.geometry.model.multivector import MultiVector

    displacement = array(displacement)
    d = displacement.shape[-1]
    dim = d + 1

    # Infer or validate metric
    if metric is None:
        metric = pga(d)
    elif metric.dim != dim:
        raise ValueError(f"Metric dim {metric.dim} doesn't match displacement dim+1 {dim}")

    if displacement.shape[-1] != dim - 1:
        raise ValueError(f"Displacement has {displacement.shape[-1]} components, expected {dim - 1}")

    # Infer collection from displacement shape if not provided
    if collection is None:
        collection = displacement.shape[:-1]

    # Scalar part: 1
    shape_0 = collection if collection else ()
    scalar_data = ones(shape_0)

    # Bivector part: (1/2) t^m e_{0m}
    shape_2 = collection + (dim, dim)
    bivector_data = zeros(shape_2)
    for m in range(1, dim):
        bivector_data[..., 0, m] = 0.5 * displacement[..., m - 1]

    components = {
        0: scalar_blade(scalar_data, metric=metric, collection=collection),
        2: Blade(data=bivector_data, grade=2, metric=metric, collection=collection),
    }

    return MultiVector(components=components, metric=metric, collection=collection)


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
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    # Import euclidean extraction from projective
    from morphis.geometry.projective import euclidean as to_euclidean

    # Validate metrics
    metric = Metric.merge(p.metric, B.metric)

    # Extract center coordinates
    c = to_euclidean(p)  # Shape: (..., d)

    # Create three components
    T1 = translator(-c, metric=metric)  # Translate to origin
    R = rotor(B, angle)  # Rotate
    T2 = translator(c, metric=metric)  # Translate back

    # Compose via geometric product: T2 @ R @ T1
    from morphis.geometry.algebra.geometric import geometric

    temp = geometric(R, T1)
    result = geometric(T2, temp)

    # Project to motor grades {0, 2}
    motor_components = {k: v for k, v in result.components.items() if k in {0, 2}}

    from numpy import broadcast_shapes

    return MultiVector(
        components=motor_components,
        metric=metric,
        collection=broadcast_shapes(p.collection, B.collection),
    )


# =============================================================================
# Screw Motion Constructor
# =============================================================================


def screw_motion(
    B: Blade,
    angle: float | NDArray,
    translation: NDArray,
    center: NDArray | None = None,
) -> MultiVector:
    """
    Create a motor for screw motion (rotation + translation along axis).

    The screw motion combines rotation in the plane defined by B with
    translation. If center is provided, rotation is about that point.

    Args:
        B: Bivector defining the rotation plane.
        angle: Rotation angle in radians.
        translation: Translation vector.
        center: Optional center point for rotation (default: origin).

    Returns:
        MultiVector (motor) representing the screw motion.

    Example:
        # Rotation in xy-plane + translation along z
        B = e1 ^ e2
        M = screw_motion(B, angle=pi/2, translation=[0, 0, 1])
        p_transformed = M @ p @ ~M
    """
    from morphis.geometry.model.multivector import MultiVector

    translation = array(translation, dtype=float)
    metric = B.metric

    from morphis.geometry.algebra.geometric import geometric

    # Create rotor and translator
    R = rotor(B, angle)
    T = translator(translation, metric=metric)

    # Compose: T @ R (translate after rotate)
    if center is not None:
        center = array(center, dtype=float)
        T_to = translator(-center, metric=metric)
        T_back = translator(center, metric=metric)

        # T_back @ T @ R @ T_to
        temp1 = geometric(R, T_to)
        temp2 = geometric(T, temp1)
        result = geometric(T_back, temp2)
    else:
        result = geometric(T, R)

    # Project to motor grades {0, 2}
    motor_components = {k: v for k, v in result.components.items() if k in {0, 2}}

    return MultiVector(
        components=motor_components,
        metric=metric,
        collection=B.collection,
    )
