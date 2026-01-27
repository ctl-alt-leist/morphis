"""
Geometric Algebra - Exponentials and Logarithms

The exponential map connects Lie algebras to Lie groups in geometric algebra.
For a vector B where B² is scalar, the exponential has a closed form:

    exp(B) = cos(√-λ) + B * sin(√-λ)/√-λ     if B² = λ < 0 (Euclidean)
    exp(B) = cosh(√λ) + B * sinh(√λ)/√λ      if B² = λ > 0 (hyperbolic)
    exp(B) = 1 + B                            if B² = 0 (nilpotent)

The logarithm is the inverse operation, extracting the generator from a versor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arctan2, cos, cosh, newaxis, ones, sin, sinh, sqrt, where, zeros
from numpy.typing import NDArray

from morphis.config import TOLERANCE
from morphis.operations.norms import norm
from morphis.operations.products import geometric, grade_project


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Vector Exponential
# =============================================================================


def exp_vector(B: Vector) -> MultiVector:
    """
    Compute the exponential of a vector: exp(B)

    For a vector B where B² is scalar (always true for simple vectors), the
    exponential has a closed form based on the sign of B²:

    - B² < 0 (Euclidean bivector): exp(B) = cos(|B|) + B * sin(|B|)/|B|
    - B² > 0 (hyperbolic): exp(B) = cosh(√λ) + B * sinh(√λ)/√λ
    - B² = 0 (nilpotent): exp(B) = 1 + B

    Args:
        B: A vector (typically bivector for rotations)

    Returns:
        MultiVector representing exp(B), typically with grades {0, B.grade}

    Examples:
        >>> from morphis.elements import Vector, euclidean_metric, basis_vectors
        >>> from math import pi
        >>> m = euclidean_metric(3)
        >>> e1, e2, e3 = basis_vectors(m)
        >>> B = (e1 ^ e2) * (pi / 4)  # 45-degree rotation plane
        >>> R = exp_vector(B)  # Rotor for 90-degree rotation
    """
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    # Handle scalar (grade-0) case: exp(s) = exp(s) * 1
    if B.grade == 0:
        from numpy import exp as np_exp

        scalar_data = np_exp(B.data)
        scalar_vector = Vector(scalar_data, grade=0, metric=B.metric, collection=B.collection)
        return MultiVector(data={0: scalar_vector}, metric=B.metric, collection=B.collection)

    # Compute B² and extract scalar part
    B_squared = geometric(B, B)
    lambda_blade = grade_project(B_squared, 0)
    lambda_val = lambda_blade.data  # Shape: (*collection,)

    # Expand lambda for broadcasting with blade data
    lambda_expanded = lambda_val
    for _ in range(B.grade):
        lambda_expanded = lambda_expanded[..., newaxis]

    # Compute coefficients based on sign of λ
    # λ < 0: trigonometric (Euclidean bivectors)
    # λ > 0: hyperbolic (Minkowski boosts)
    # λ ≈ 0: nilpotent

    # Trigonometric case: λ < 0, so √(-λ) is real
    # cos(√(-λ)) + B * sin(√(-λ))/√(-λ)
    sqrt_neg_lambda = sqrt(where(lambda_val < -TOLERANCE, -lambda_val, 1.0))
    trig_scalar = cos(sqrt_neg_lambda)
    trig_blade_coeff = where(
        abs(sqrt_neg_lambda) > TOLERANCE,
        sin(sqrt_neg_lambda) / sqrt_neg_lambda,
        ones(lambda_val.shape),  # sinc(0) = 1
    )

    # Hyperbolic case: λ > 0
    sqrt_pos_lambda = sqrt(where(lambda_val > TOLERANCE, lambda_val, 1.0))
    hyp_scalar = cosh(sqrt_pos_lambda)
    hyp_blade_coeff = where(
        abs(sqrt_pos_lambda) > TOLERANCE,
        sinh(sqrt_pos_lambda) / sqrt_pos_lambda,
        ones(lambda_val.shape),  # sinhc(0) = 1
    )

    # Select based on sign of λ
    is_negative = lambda_val < -TOLERANCE
    is_positive = lambda_val > TOLERANCE

    # Final coefficients
    scalar_coeff = where(is_negative, trig_scalar, where(is_positive, hyp_scalar, ones(lambda_val.shape)))

    blade_coeff = where(is_negative, trig_blade_coeff, where(is_positive, hyp_blade_coeff, ones(lambda_val.shape)))

    # Expand blade_coeff for broadcasting
    blade_coeff_expanded = blade_coeff
    for _ in range(B.grade):
        blade_coeff_expanded = blade_coeff_expanded[..., newaxis]

    # Build result components
    scalar_vector = Vector(scalar_coeff, grade=0, metric=B.metric, collection=B.collection)
    scaled_B = Vector(
        blade_coeff_expanded * B.data,
        grade=B.grade,
        metric=B.metric,
        collection=B.collection,
    )

    return MultiVector(
        data={0: scalar_vector, B.grade: scaled_B},
        metric=B.metric,
        collection=B.collection,
    )


# =============================================================================
# Versor Logarithm
# =============================================================================


def log_versor(M: MultiVector) -> Vector:
    """
    Extract the bivector generator from a rotor/versor: log(M)

    For a rotor M = a + B where a is scalar and B is bivector:

        log(M) = atan2(|B|, a) * B / |B|

    The result is a bivector whose exponential recovers M.

    Args:
        M: MultiVector representing a rotor (typically grades {0, 2})

    Returns:
        Vector (bivector) that generates M via exp_vector

    Raises:
        ValueError: If M doesn't have the expected rotor structure

    Notes:
        - For rotors near identity (a ≈ 1, |B| ≈ 0), uses careful numerics
        - Returns angle in (-pi, pi]
        - For double-cover rotors, may return either generator

    Examples:
        >>> R = rotor(B, angle)
        >>> B_recovered = log_versor(R)
        >>> # B_recovered ≈ -B * angle / 2
    """
    from morphis.elements.vector import Vector

    # Extract scalar and bivector parts
    scalar_part = M.grade_select(0)
    bivector_part = M.grade_select(2)

    if scalar_part is None:
        raise ValueError("Rotor must have grade-0 (scalar) component")

    if bivector_part is None:
        # Pure scalar: log(a) = 0 bivector (identity rotor)
        d = M.dim
        shape = M.collection + (d, d)
        return Vector(zeros(shape), grade=2, metric=M.metric, collection=M.collection)

    a = scalar_part.data  # Shape: (*collection,)
    B_norm = norm(bivector_part)  # Shape: (*collection,)

    # Compute angle: theta = 2 * atan2(|B|, a) for rotor R = exp(-B*theta/2)
    # But log returns the generator, so we want atan2(|B|, a) directly
    theta = arctan2(B_norm, a)

    # Scale bivector: result = theta * B / |B|
    # Handle |B| ≈ 0 case (near identity)
    safe_norm = where(B_norm > TOLERANCE, B_norm, 1.0)
    scale = where(B_norm > TOLERANCE, theta / safe_norm, zeros(B_norm.shape))

    # Expand for broadcasting
    scale_expanded = scale
    for _ in range(2):  # grade 2
        scale_expanded = scale_expanded[..., newaxis]

    result_data = scale_expanded * bivector_part.data

    return Vector(result_data, grade=2, metric=M.metric, collection=M.collection)


# =============================================================================
# Spherical Linear Interpolation
# =============================================================================


def slerp(R0: MultiVector, R1: MultiVector, t: float | NDArray) -> MultiVector:
    """
    Spherical linear interpolation between rotors.

    Computes R(t) that smoothly interpolates from R0 (at t=0) to R1 (at t=1)
    along the geodesic path on the rotation manifold.

        R(t) = R0 * exp(t * log(R0^(-1) * R1))

    This produces constant angular velocity interpolation without the
    artifacts of linear interpolation in rotor components.

    Args:
        R0: Starting rotor (MultiVector with grades {0, 2})
        R1: Ending rotor (MultiVector with grades {0, 2})
        t: Interpolation parameter, 0 <= t <= 1 (can be array for batch)

    Returns:
        Interpolated rotor at parameter t

    Examples:
        >>> R0 = rotor(e12, 0)  # Identity
        >>> R1 = rotor(e12, pi / 2)  # 90-degree rotation
        >>> R_mid = slerp(R0, R1, 0.5)  # 45-degree rotation
    """
    from numpy import asarray

    from morphis.elements.vector import Vector

    t = asarray(t)

    # Compute relative rotor: R_rel = R0^(-1) * R1
    R0_inv = R0.inverse()
    R_rel = R0_inv * R1

    # Extract generator bivector
    B = log_versor(R_rel)

    # Scale by t, handling array t values
    # Need to expand t for proper broadcasting with bivector shape
    if t.ndim == 0:
        # Scalar t
        B_scaled = B * float(t)
    else:
        # Array t: need to create batch of scaled bivectors
        # t shape: (N,) -> expand to (N, 1, 1) for broadcasting with (d, d)
        t_expanded = t
        for _ in range(B.grade):
            t_expanded = t_expanded[..., newaxis]

        B_scaled = Vector(
            t_expanded * B.data,
            grade=B.grade,
            metric=B.metric,
            collection=t.shape + B.collection,
        )

    # Compute exp(t * B)
    exp_tB = exp_vector(B_scaled)

    # Compose: R(t) = R0 * exp(t * B)
    return R0 * exp_tB
