"""
Geometric Algebra - Rotation and Similarity Constructors

Constructor functions for rotors and similarity versors that return MultiVector
objects. These work with the geometric product operator `*` for transformations
via sandwich products: transformed = M * b * ~M

All operations support collection dimensions via einsum broadcasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array, broadcast_shapes, newaxis, sqrt as np_sqrt

from morphis.config import TOLERANCE
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


# =============================================================================
# Similarity Versor Constructor
# =============================================================================


def align_vectors(u: Vector, v: Vector) -> MultiVector:
    """
    Create a similarity versor that transforms u to v.

    Returns S such that: transform(u, S) = v (exactly, including magnitude).

    S = √(|v|/|u|) · R, where R is the rotor aligning directions.

    The similarity versor is a MultiVector with grades {0, 2}. Apply via
    sandwich product: transformed = S * b * ~S

    Properties:
        - S · ~S = |v|/|u| (the scale factor, not 1)
        - Composition: S₂ · S₁ applies S₁ then S₂
        - transform(u, S) = v (by construction)

    Args:
        u: Source vector (grade-1). The vector to transform from.
        v: Target vector (grade-1). The vector to transform to.

    Returns:
        MultiVector (similarity versor) with grades {0, 2}.

    Example:
        u = Vector([1, 0, 0], grade=1, metric=g)
        v = Vector([0, 2, 0], grade=1, metric=g)
        S = align_vectors(u, v)
        w = transform(u, S)  # w ≈ v
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector
    from morphis.operations.norms import form, norm, unit

    if u.grade != 1 or v.grade != 1:
        raise ValueError(f"align_vectors requires grade-1 vectors, got grades {u.grade} and {v.grade}")

    metric = Metric.merge(u.metric, v.metric)

    # Compute scale factor: √(|v|/|u|)
    norm_u = norm(u)
    norm_v = norm(v)
    scale = np_sqrt(norm_v / norm_u)

    # Unit vectors
    u_hat = unit(u)
    v_hat = unit(v)

    # Rotor aligning u_hat to v_hat: R = normalize(v_hat * u_hat + 1)
    # For unit vectors: v̂û = cos(θ) + sin(θ)B̂
    # So v̂û + 1 = (1 + cos(θ)) + sin(θ)B̂ ∝ R
    vu = geometric(v_hat, u_hat)

    # Add 1 to scalar part
    scalar_part = vu.grade_select(0)
    if scalar_part is not None:
        new_scalar = scalar_part.data + 1.0
    else:
        new_scalar = 1.0

    # Build unnormalized rotor
    rotor_components = {}
    rotor_components[0] = Vector(
        data=new_scalar,
        grade=0,
        metric=metric,
    )
    bivector_part = vu.grade_select(2)
    if bivector_part is not None:
        rotor_components[2] = bivector_part

    r_unnorm = MultiVector(data=rotor_components, metric=metric)

    # Compute norm for normalization
    # |v̂û + 1|² = R · ~R = (1 + cos(θ))² + sin²(θ) = 2(1 + cos(θ))
    r_norm_sq = form(r_unnorm)
    r_norm = np_sqrt(r_norm_sq)

    # Handle antiparallel case: when r_norm ≈ 0, vectors are antiparallel (θ = π)
    # In this case, we need to rotate 180° in any plane containing u
    # R = B̂ where B̂ is any unit bivector in a plane containing u
    is_antiparallel = r_norm < TOLERANCE

    if is_antiparallel.any() if hasattr(is_antiparallel, "any") else is_antiparallel:
        # Find perpendicular vector by trying cross with basis vectors
        from morphis.operations.products import wedge

        d = u.dim

        # Try wedging with each basis vector until we get nonzero
        for i in range(d):
            basis = Vector(
                data=array([1.0 if j == i else 0.0 for j in range(d)]),
                grade=1,
                metric=metric,
            )
            B = wedge(u_hat, basis)
            B_norm = norm(B)
            if (B_norm > TOLERANCE).all() if hasattr(B_norm, "all") else B_norm > TOLERANCE:
                break

        B_hat = unit(B)

        # For 180° rotation, R = B̂ (pure bivector)
        antiparallel_rotor = MultiVector(
            data={2: B_hat},
            metric=metric,
        )

        # Blend based on antiparallel mask
        if hasattr(is_antiparallel, "__len__"):
            # Batch case - would need more sophisticated blending
            # For now, handle scalar case
            pass

        if is_antiparallel if not hasattr(is_antiparallel, "any") else is_antiparallel.all():
            # All antiparallel - use antiparallel rotor
            r = antiparallel_rotor
        else:
            # Normalize the regular rotor
            r = unit(r_unnorm)
    else:
        # Normalize to get unit rotor
        r = unit(r_unnorm)

    # Scale the rotor by √scale to get similarity versor
    # S = √(|v|/|u|) · R
    # Expand scale for broadcasting over grade components
    similarity_components = {}
    for grade_k, component in r.data.items():
        scale_expanded = scale
        for _ in range(grade_k):
            scale_expanded = scale_expanded[..., newaxis]
        similarity_components[grade_k] = Vector(
            data=scale_expanded * component.data,
            grade=grade_k,
            metric=metric,
        )

    return MultiVector(data=similarity_components, metric=metric)


def point_alignment(
    u1: Vector,
    u2: Vector,
    v1: Vector,
    v2: Vector,
) -> tuple[MultiVector, Vector]:
    """
    Compute similarity transform aligning two point pairs.

    Given source points (u1, u2) and target points (v1, v2), computes the
    similarity versor S and translation t such that:

        transform(u1, S) + t = v1
        transform(u2, S) + t = v2

    The axis u2 - u1 is aligned to v2 - v1 (direction and scale), then
    translated so u1 maps to v1.

    Args:
        u1: First source point (grade-1 vector).
        u2: Second source point (grade-1 vector).
        v1: First target point (grade-1 vector).
        v2: Second target point (grade-1 vector).

    Returns:
        Tuple of (S, t) where:
            S: Similarity versor (MultiVector). Apply via sandwich product.
            t: Translation vector (Vector). Add after sandwich product.

        To transform a point p: p' = transform(p, S) + t

    Example:
        # Register local model coordinates to world coordinates
        S, t = point_alignment(p1_local, p2_local, p1_world, p2_world)

        # Transform all vertices
        for v in vertices:
            v_world = transform(v, S) + t
    """
    from morphis.transforms.actions import transform

    if any(p.grade != 1 for p in [u1, u2, v1, v2]):
        raise ValueError("point_alignment requires grade-1 vectors")

    # Compute axis vectors
    axis_u = u2 - u1
    axis_v = v2 - v1

    # Get similarity versor aligning the axes
    s = align_vectors(axis_u, axis_v)

    # Compute translation: t = v1 - transform(u1, S)
    u1_transformed = transform(u1, s)
    t = v1 - u1_transformed

    return s, t
