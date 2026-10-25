"""
Geometric Algebra - Exponentials and Logarithms

The exponential map connects Lie algebras to Lie groups in geometric algebra.
For a vector B where B² is scalar, the exponential has a closed form:

    exp(B) = cos(√-λ) + B * sin(√-λ)/√-λ     if B² = λ < 0 (Euclidean)
    exp(B) = cosh(√λ) + B * sinh(√λ)/√λ      if B² = λ > 0 (hyperbolic)
    exp(B) = 1 + B                            if B² = 0 (nilpotent)

A bivector in four or five dimensions need not square to a scalar: it is the
sum B = B₁ + B₂ of two commuting simple bivectors in orthogonal planes, and
B² = λ₁ + λ₂ + 2 W with W = ½ B ∧ B = B₁ B₂. Its exponential is the product
exp(B₁) exp(B₂), written through B and W alone (the invariant decomposition of
Roelfs and De Keninck), so the split itself is never formed.

The logarithm is the inverse operation, extracting the generator from a versor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import (
    abs as np_abs,
    arctan2,
    cos,
    cosh,
    iscomplexobj,
    newaxis,
    ones,
    pi,
    sin,
    sinc,
    sinh,
    sqrt,
    where,
    zeros,
)
from numpy.typing import NDArray

from morphis.config import TOLERANCE
from morphis.operations.norms import norm
from morphis.operations.products import geometric, grade_project, wedge


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Vector Exponential
# =============================================================================


def exp_vector(B: Vector) -> MultiVector:
    """
    Compute the exponential of a vector: exp(B)

    A bivector of a space of dimension 4 or 5 may be non-simple, the sum of
    two rotations, a rotation and a boost, or a screw in PGA; its exponential
    is exact in closed form for every metric (see exp_bivector). In six or
    more dimensions a bivector must be simple.

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

    if B.grade == 2:
        return exp_bivector(B)

    return _exp_scalar_square(B)


def _exp_scalar_square(B: Vector) -> MultiVector:
    """The closed form for a vector B whose square is a scalar λ: cos, cosh, or 1 + B by the sign of λ."""
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

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
# Bivector Exponential
# =============================================================================


def exp_bivector(B: Vector) -> MultiVector:
    """
    Exponential of a bivector that need not be simple: exp(B).

    In four or five dimensions B = B₁ + B₂, with B₁ and B₂ simple, commuting,
    and in orthogonal planes, so B_k² = λ_k are scalars and B₁ B₂ = B₁ ∧ B₂.
    The two invariants of B determine the λ_k:

        ⟨B²⟩₀ = λ₁ + λ₂,    W = ½ B ∧ B = B₁ B₂,    W² = λ₁ λ₂

    so they are the roots of λ² - ⟨B²⟩₀ λ + W² = 0, and B_k = (λ_k B - B W) / (λ_k - λ_l).
    With c_k = cos √-λ_k and s_k = sin √-λ_k / √-λ_k, entire functions of λ_k
    that are cosh and sinh for λ_k > 0 and 1 at λ_k = 0, the product
    exp(B₁) exp(B₂) is

        c₁ c₂ + s₁ s₂ W + ½ (c₂ s₁ + c₁ s₂) B + ½ D (⟨B²⟩₀ B - 2 B W)

    where D = (c₂ s₁ - c₁ s₂) / (λ₁ - λ₂) is a divided difference, finite as
    the roots meet. One formula covers rotations, boosts, and null
    translations together, and a simple bivector (W = 0) reduces to
    cos + B sin. When the roots coincide, as for an isoclinic rotation, any
    split serves and ⟨B²⟩₀ B - 2 B W vanishes, so D is taken at its limit.

    Args:
        B: Bivector of any lot; in six or more dimensions it must be simple

    Outside four and five dimensions B ∧ B is zero (below four) or must be
    (six and above), and the simple closed form cos + B sin applies.

    Returns:
        MultiVector with grades {0, 2, 4}, or {0, 2} for a simple bivector
        outside four and five dimensions

    Raises:
        NotImplementedError: For a non-simple bivector in six or more dimensions,
            which has three or more invariant planes
    """
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    if B.dim < 4:
        return _exp_scalar_square(B)

    W = 0.5 * wedge(B, B)

    if B.dim >= 6:
        scale = norm(B) ** 2 + 1.0
        is_simple = (norm(W) <= 1e-9 * scale).all()
        if not is_simple:
            raise NotImplementedError(
                f"exp of a non-simple bivector in {B.dim} dimensions needs three or more invariant planes"
            )

        return _exp_scalar_square(B)

    metric, collection = B.metric, B.collection
    square = grade_project(geometric(B, B), 0).data
    product = grade_project(geometric(W, W), 0).data

    # Roots of λ² - square λ + product, the larger one first so the smaller keeps its precision
    half = 0.5 * square + 0j
    root = sqrt(half**2 - product)
    is_plus = np_abs(half + root) >= np_abs(half - root)
    large = where(is_plus, half + root, half - root)
    small = where(np_abs(large) > 0, product / where(np_abs(large) > 0, large, 1.0), 0.0)

    c_large, s_large = _even_cos(large), _even_sinc(large)
    c_small, s_small = _even_cos(small), _even_sinc(small)

    gap = large - small
    is_split = np_abs(gap) > 1e-6 * (1.0 + np_abs(half))
    difference = (c_small * s_large - c_large * s_small) / where(is_split, gap, 1.0)
    divided = where(is_split, difference, _divided_limit(half))

    scalar = c_large * c_small
    along_B = 0.5 * (c_small * s_large + c_large * s_small) + 0.5 * divided * square
    along_BW = -divided
    along_W = s_large * s_small

    BW = grade_project(geometric(B, W), 2).data

    def components(value: NDArray, grade: int) -> NDArray:
        expanded = value
        for _ in range(grade):
            expanded = expanded[..., newaxis]

        return expanded if iscomplexobj(B.data) else expanded.real

    bivector = components(along_B, 2) * B.data + components(along_BW, 2) * BW
    data = {
        0: Vector(components(scalar, 0), grade=0, metric=metric, collection=collection),
        2: Vector(bivector, grade=2, metric=metric, collection=collection),
        4: Vector(components(along_W, 4) * W.data, grade=4, metric=metric, collection=collection),
    }

    return MultiVector(data=data, metric=metric, collection=collection)


def _even_cos(square: NDArray) -> NDArray:
    """cos √-λ as a function of λ: cosh √λ for λ > 0, 1 at 0."""
    return cos(sqrt(-square + 0j))


def _even_sinc(square: NDArray) -> NDArray:
    """sin √-λ / √-λ as a function of λ: sinh √λ / √λ for λ > 0, 1 at 0."""
    return sinc(sqrt(-square + 0j) / pi)


def _divided_limit(mean: NDArray) -> NDArray:
    """
    The divided difference D at coincident roots λ₁ = λ₂ = μ: c s' - c' s at μ.

    With c' = s / 2 and s' = (c - s) / (2 μ), and the series 1/6 + μ/60 + μ²/2520
    + μ³/181440 for s' near μ = 0.
    """
    c, s = _even_cos(mean), _even_sinc(mean)
    is_small = np_abs(mean) < 1e-3
    safe = where(is_small, 1.0, mean)
    derivative = where(is_small, 1 / 6 + mean / 60 + mean**2 / 2520 + mean**3 / 181440, (c - s) / (2 * safe))

    return c * derivative - 0.5 * s * s


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
