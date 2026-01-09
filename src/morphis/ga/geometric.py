"""
Geometric Algebra - Geometric Product

The geometric product unifies the dot and wedge products into a single
associative operation. For vectors u and v:

    uv = u . v + u ^ v

This extends to all grades through systematic contraction and antisymmetrization.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from numpy import einsum, newaxis, zeros
from numpy.typing import NDArray

from morphis.ga.context import GeometricContext
from morphis.ga.model import Blade, Metric, MultiVector, euclidean, scalar_blade
from morphis.ga.structure import (
    generalized_delta,
    geometric_normalization,
    geometric_signature,
)
from morphis.ga.utils import broadcast_collection_shape, get_common_cdim, get_common_dim


# =============================================================================
# Geometric Product
# =============================================================================


def _geometric_bl_bl(u: Blade, v: Blade, g: Metric | None = None) -> MultiVector:
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


def grade_project(M: MultiVector, k: int) -> Blade:
    """
    Extract grade-k component from multivector: <M>_k

    Returns grade-k blade if present, otherwise zero blade.
    """
    component = M.grade_select(k)

    if component is not None:
        return component

    # Return zero blade of appropriate grade
    d = M.dim
    cdim = M.cdim
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
    M = geometric(u, v, g)
    s = M.grade_select(0)

    if s is not None:
        return s.data

    # No scalar component
    return zeros(broadcast_collection_shape(u, v))


def commutator(u: Blade, v: Blade, g: Metric | None = None) -> MultiVector:
    """
    Commutator product: [u, v] = (1 / 2) (uv - vu)

    Extracts antisymmetric part (odd grade differences).
    """
    uv = geometric(u, v, g)
    vu = geometric(v, u, g)

    return 0.5 * (uv - vu)


def anticommutator(u: Blade, v: Blade, g: Metric | None = None) -> MultiVector:
    """
    Anticommutator product: u * v = (1 / 2) (uv + vu)

    Extracts symmetric part (even grade differences).
    """
    uv = geometric(u, v, g)
    vu = geometric(v, u, g)

    return 0.5 * (uv + vu)


# =============================================================================
# Reversion and Inverse
# =============================================================================


def _reverse_bl(u: Blade) -> Blade:
    """
    Reverse a blade: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k blade.

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


def _reverse_mv(M: MultiVector) -> MultiVector:
    """
    Reverse each component of a multivector.

    Returns MultiVector with all components reversed.
    """
    components = {k: _reverse_bl(blade) for k, blade in M.components.items()}

    return MultiVector(components=components, dim=M.dim, cdim=M.cdim)


def reverse(u: Blade | MultiVector) -> Blade | MultiVector:
    """
    Reverse: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k blade.

    For multivectors, reverses each component.

    Reverses the order of vector factors.
    Context: Preserves input context.
    """
    if isinstance(u, Blade):
        return _reverse_bl(u)

    return _reverse_mv(u)


def _inverse_bl(u: Blade, g: Metric | None = None) -> Blade:
    """
    Inverse of a blade: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    Context: Preserves input blade context.
    """
    g = euclidean(u.dim) if g is None else g

    u_rev = _reverse_bl(u)
    u_u_rev = _geometric_bl_bl(u, u_rev, g)

    # Extract scalar part
    s = u_u_rev.grade_select(0)
    if s is None:
        raise ValueError("Blade square is not scalar - cannot invert")

    # Divide reversed blade by scalar
    s_expanded = s.data
    for _ in range(u.grade):
        s_expanded = s_expanded[..., newaxis]

    return Blade(
        data=u_rev.data / s_expanded,
        grade=u.grade,
        dim=u.dim,
        cdim=u.cdim,
        context=u.context,
    )


def _inverse_mv(M: MultiVector, g: Metric | None = None) -> MultiVector:
    """
    Inverse of a multivector: M^(-1) = reverse(M) / (M * reverse(M))

    Requires M * reverse(M) to be invertible scalar.
    """
    g = euclidean(M.dim) if g is None else g

    M_rev = _reverse_mv(M)
    M_M_rev = _geometric_mv_mv(M, M_rev, g)

    # Extract scalar part
    s = M_M_rev.grade_select(0)
    if s is None:
        raise ValueError("MultiVector product with reverse is not scalar - cannot invert")

    return M_rev * (1.0 / s.data)


def inverse(u: Blade | MultiVector, g: Metric | None = None) -> Blade | MultiVector:
    """
    Inverse: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    Context: Preserves input context.
    """
    if isinstance(u, Blade):
        return _inverse_bl(u, g)

    return _inverse_mv(u, g)


def _geometric_mv_mv(M: MultiVector, N: MultiVector, g: Metric | None = None) -> MultiVector:
    """
    Geometric product of two multivectors.

    Computes all pairwise geometric products of components and sums.
    """
    g = euclidean(M.dim) if g is None else g

    if M.dim != N.dim:
        raise ValueError(f"Dimension mismatch: {M.dim} != {N.dim}")

    result_components: dict[int, Blade] = {}

    for _k1, u in M.components.items():
        for _k2, v in N.components.items():
            product = _geometric_bl_bl(u, v, g)
            for grade, component in product.components.items():
                if grade in result_components:
                    result_components[grade] = result_components[grade] + component
                else:
                    result_components[grade] = component

    return MultiVector(components=result_components, dim=M.dim, cdim=max(M.cdim, N.cdim))


def geometric(u: Blade | MultiVector, v: Blade | MultiVector, g: Metric | None = None) -> MultiVector:
    """
    Geometric product of two blades or multivectors: uv = sum of <uv>_r over grades r

    For blades of grade j and k, produces components at grades
    |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).

    Each grade r corresponds to (j + k - r) / 2 metric contractions.

    Context: Preserves if both match, otherwise None.

    Returns MultiVector containing all nonzero grade components.
    """
    # Both blades
    if isinstance(u, Blade) and isinstance(v, Blade):
        return _geometric_bl_bl(u, v, g)

    # Convert blades to multivectors if needed
    if isinstance(u, Blade):
        u = MultiVector(components={u.grade: u}, dim=u.dim, cdim=u.cdim)

    if isinstance(v, Blade):
        v = MultiVector(components={v.grade: v}, dim=v.dim, cdim=v.cdim)

    return _geometric_mv_mv(u, v, g)


# =============================================================================
# Geometric Product with Mixed Types (for operators)
# =============================================================================


def geometric_bl_mv(u: Blade, M: MultiVector, g: Metric | None = None) -> MultiVector:
    """
    Geometric product of blade with multivector: u @ M

    Distributes over components.

    Context: Preserves if all match, otherwise None.

    Returns MultiVector.
    """
    g = euclidean(u.dim) if g is None else g
    result_components: dict[int, Blade] = {}

    for _k, component in M.components.items():
        product = _geometric_bl_bl(u, component, g)

        for grade, blade in product.components.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + blade
            else:
                result_components[grade] = blade

    return MultiVector(
        components=result_components,
        dim=u.dim,
        cdim=max(u.cdim, M.cdim),
    )


def geometric_mv_bl(M: MultiVector, u: Blade, g: Metric | None = None) -> MultiVector:
    """
    Geometric product of multivector with blade: M @ u

    Distributes over components.

    Context: Preserves if all match, otherwise None.

    Returns MultiVector.
    """
    g = euclidean(M.dim) if g is None else g
    result_components: dict[int, Blade] = {}

    for _k, component in M.components.items():
        product = _geometric_bl_bl(component, u, g)

        for grade, blade in product.components.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + blade
            else:
                result_components[grade] = blade

    return MultiVector(
        components=result_components,
        dim=M.dim,
        cdim=max(M.cdim, u.cdim),
    )
