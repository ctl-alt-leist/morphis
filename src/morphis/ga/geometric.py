"""
Geometric Algebra - Geometric Product

The geometric product unifies the dot and wedge products into a single
associative operation. For vectors u and v:

    uv = u . v + u ^ v

This extends to all grades through systematic contraction and antisymmetrization.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from numpy import einsum, zeros
from numpy.typing import NDArray

from morphis.ga.context import GeometricContext
from morphis.ga.model import Blade, Metric, MultiVector, euclidean, scalar_blade
from morphis.ga.structure import (
    generalized_delta,
    geometric_normalization,
    geometric_signature,
)
from morphis.ga.utils import get_common_cdim, get_common_dim


# =============================================================================
# Geometric Product
# =============================================================================


def geometric(u: Blade, v: Blade, g: Metric | None = None) -> MultiVector:
    """
    Geometric product of two blades: uv = sum of <uv>_r over grades r

    For blades of grade j and k, produces components at grades
    |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).

    Each grade r corresponds to (j + k - r)/2 metric contractions.

    Context: Preserves if both match, otherwise None.

    Returns MultiVector containing all nonzero grade components.
    """
    d = get_common_dim(u, v)
    g = euclidean(d) if g is None else g

    if d != g.dim:
        raise ValueError(f"Blade dim {d} != metric dim {g.dim}")

    j, k = u.grade, v.grade
    cdim = get_common_cdim(u, v)
    merged_context = GeometricContext.merge(u.context, v.context)

    components = {}

    # Compute all grade components
    min_grade = abs(j - k)
    max_grade = min(j + k, d)

    for r in range(min_grade, max_grade + 1, 2):
        # Number of contractions for this grade
        c = (j + k - r) // 2

        if c < 0 or c > min(j, k):
            continue

        # Build einsum signature and compute
        sig = geometric_signature(j, k, c)
        norm = geometric_normalization(j, k, c)

        # Prepare arguments for einsum
        metric_args = [g.data] * c

        if r == 0:
            # Scalar result
            result_data = norm * einsum(sig, *metric_args, u.data, v.data)
            component = scalar_blade(result_data, dim=d, cdim=cdim)
        elif c == 0:
            # Pure wedge (no contractions)
            delta = generalized_delta(r, d)
            result_data = norm * einsum(sig, u.data, v.data, delta)
            component = Blade(data=result_data, grade=r, dim=d, cdim=cdim)
        else:
            # Mixed: contractions + antisymmetrization
            delta = generalized_delta(r, d)
            result_data = norm * einsum(sig, *metric_args, u.data, v.data, delta)
            component = Blade(data=result_data, grade=r, dim=d, cdim=cdim)

        component.context = merged_context
        components[r] = component

    return MultiVector(components=components, dim=d, cdim=cdim)


# =============================================================================
# Grade Projection
# =============================================================================


def grade_project(m: MultiVector, k: int) -> Blade:
    """
    Extract grade-k component from multivector: <M>_k

    Returns grade-k blade if present, otherwise zero blade.
    """
    component = m.grade_select(k)

    if component is not None:
        return component

    # Return zero blade of appropriate grade
    d = m.dim
    cdim = m.cdim
    shape = (1,) * cdim + (d,) * k if k > 0 else (1,) * cdim
    return Blade(data=zeros(shape), grade=k, dim=d, cdim=cdim)


# =============================================================================
# Component Products
# =============================================================================


def scalar_product(u: Blade, v: Blade, g: Metric | None = None) -> NDArray:
    """
    Scalar part of geometric product: <uv>_0

    Returns scalar array with shape collection_shape.
    """
    mv = geometric(u, v, g)
    scalar_blade = mv.grade_select(0)

    if scalar_blade is not None:
        return scalar_blade.data

    # No scalar component
    from morphis.ga.utils import broadcast_collection_shape

    return zeros(broadcast_collection_shape(u, v))


def commutator(u: Blade, v: Blade, g: Metric | None = None) -> MultiVector:
    """
    Commutator product: [u, v] = (1/2)(uv - vu)

    Extracts antisymmetric part (odd grade differences).
    """
    uv = geometric(u, v, g)
    vu = geometric(v, u, g)
    return (uv - vu) * 0.5


def anticommutator(u: Blade, v: Blade, g: Metric | None = None) -> MultiVector:
    """
    Anticommutator product: u * v = (1/2)(uv + vu)

    Extracts symmetric part (even grade differences).
    """
    uv = geometric(u, v, g)
    vu = geometric(v, u, g)
    return (uv + vu) * 0.5


# =============================================================================
# Reversion and Inverse
# =============================================================================


def reverse(u: Blade) -> Blade:
    """
    Reverse: reverse(u) = (-1)^(k(k-1)/2) u for grade-k blade.

    Reverses the order of vector factors in the blade.
    Context: Preserves input blade context.
    """
    k = u.grade
    sign = (-1) ** (k * (k - 1) // 2)

    return Blade(
        data=sign * u.data,
        grade=k,
        dim=u.dim,
        cdim=u.cdim,
        context=u.context,
    )


def reverse_mv(m: MultiVector) -> MultiVector:
    """
    Reverse each component of a multivector.

    Returns MultiVector with all components reversed.
    """
    components = {k: reverse(blade) for k, blade in m.components.items()}
    return MultiVector(components=components, dim=m.dim, cdim=m.cdim)


def inverse(u: Blade, g: Metric | None = None) -> Blade:
    """
    Inverse: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    Context: Preserves input blade context.
    """
    g = euclidean(u.dim) if g is None else g

    u_rev = reverse(u)
    u_u_rev = geometric(u, u_rev, g)

    # Extract scalar part
    scalar = u_u_rev.grade_select(0)
    if scalar is None:
        raise ValueError("Blade square is not scalar - cannot invert")

    # Divide reversed blade by scalar
    from numpy import newaxis

    scalar_expanded = scalar.data
    for _ in range(u.grade):
        scalar_expanded = scalar_expanded[..., newaxis]

    return Blade(
        data=u_rev.data / scalar_expanded,
        grade=u.grade,
        dim=u.dim,
        cdim=u.cdim,
        context=u.context,
    )


def geometric_mv_mv(m1: MultiVector, m2: MultiVector, g: Metric | None = None) -> MultiVector:
    """
    Geometric product of two multivectors.

    Computes all pairwise geometric products of components and sums.
    """
    g = euclidean(m1.dim) if g is None else g

    if m1.dim != m2.dim:
        raise ValueError(f"Dimension mismatch: {m1.dim} != {m2.dim}")

    result_components: dict[int, Blade] = {}

    for _k1, blade1 in m1.components.items():
        for _k2, blade2 in m2.components.items():
            product = geometric(blade1, blade2, g)
            for grade, component in product.components.items():
                if grade in result_components:
                    result_components[grade] = result_components[grade] + component
                else:
                    result_components[grade] = component

    return MultiVector(components=result_components, dim=m1.dim, cdim=max(m1.cdim, m2.cdim))


def inverse_mv(m: MultiVector, g: Metric | None = None) -> MultiVector:
    """
    Inverse of a multivector: m^(-1) = reverse(m) / (m * reverse(m))

    Requires m * reverse(m) to be invertible scalar.
    """
    g = euclidean(m.dim) if g is None else g

    m_rev = reverse_mv(m)
    m_m_rev = geometric_mv_mv(m, m_rev, g)

    # Extract scalar part
    scalar = m_m_rev.grade_select(0)
    if scalar is None:
        raise ValueError("MultiVector product with reverse is not scalar - cannot invert")

    # Divide reversed multivector by scalar
    return m_rev * (1.0 / scalar.data)
