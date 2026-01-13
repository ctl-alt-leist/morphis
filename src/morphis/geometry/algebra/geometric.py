"""
Geometric Algebra - Geometric Product

The geometric product unifies the dot and wedge products into a single
associative operation. For vectors u and v:

    uv = u . v + u ^ v

This extends to all grades through systematic contraction and antisymmetrization.
Metrics are obtained from blade attributes and validated for compatibility.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import einsum, newaxis, zeros
from numpy.typing import NDArray

from morphis.geometry.algebra.structure import (
    generalized_delta,
    geometric_normalization,
    geometric_signature,
)
from morphis.geometry.utils import broadcast_collection_shape, get_common_dim


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.multivector import MultiVector


# =============================================================================
# Geometric Product (Blade x Blade)
# =============================================================================


def _geometric_bl_bl(u: Blade, v: Blade) -> MultiVector:
    """
    Geometric product of two blades: uv = sum of <uv>_r over grades r

    For blades of grade j and k, produces components at grades
    |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).

    Each grade r corresponds to (j + k - r)/2 metric contractions.

    Both blades must have compatible metrics (validated via Metric.merge).

    Returns MultiVector containing all nonzero grade components.
    """
    from morphis.geometry.model.blade import Blade, scalar_blade
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    d = get_common_dim(u, v)
    j, k = u.grade, v.grade

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
            component = scalar_blade(result_data, metric=metric)

        elif c == 0:
            # Pure wedge (no contractions)
            delta = generalized_delta(r, d)
            result_data = norm * einsum(sig, u.data, v.data, delta)
            component = Blade(data=result_data, grade=r, metric=metric)

        else:
            # Mixed: contractions + antisymmetrization
            delta = generalized_delta(r, d)
            result_data = norm * einsum(sig, *metric_args, u.data, v.data, delta)
            component = Blade(data=result_data, grade=r, metric=metric)

        components[r] = component

    return MultiVector(components=components, metric=metric)


# =============================================================================
# Geometric Product (MultiVector x MultiVector)
# =============================================================================


def _geometric_mv_mv(M: MultiVector, N: MultiVector) -> MultiVector:
    """
    Geometric product of two multivectors.

    Computes all pairwise geometric products of components and sums.

    Both multivectors must have compatible metrics.
    """
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(M.metric, N.metric)

    result_components: dict[int, Blade] = {}

    for _k1, u in M.components.items():
        for _k2, v in N.components.items():
            product = _geometric_bl_bl(u, v)
            for grade, component in product.components.items():
                if grade in result_components:
                    result_components[grade] = result_components[grade] + component
                else:
                    result_components[grade] = component

    return MultiVector(components=result_components, metric=metric)


# =============================================================================
# Public Interface
# =============================================================================


def geometric(u: Blade | MultiVector, v: Blade | MultiVector) -> MultiVector:
    """
    Geometric product of two blades or multivectors: uv = sum of <uv>_r over grades r

    For blades of grade j and k, produces components at grades
    |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).

    Each grade r corresponds to (j + k - r) / 2 metric contractions.

    Both operands must have compatible metrics (validated via Metric.merge).

    Returns MultiVector containing all nonzero grade components.
    """
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.multivector import MultiVector

    # Both blades
    if isinstance(u, Blade) and isinstance(v, Blade):
        return _geometric_bl_bl(u, v)

    # Convert blades to multivectors if needed
    if isinstance(u, Blade):
        u = MultiVector(components={u.grade: u}, metric=u.metric)

    if isinstance(v, Blade):
        v = MultiVector(components={v.grade: v}, metric=v.metric)

    return _geometric_mv_mv(u, v)


# =============================================================================
# Grade Projection
# =============================================================================


def grade_project(M: MultiVector, k: int) -> Blade:
    """
    Extract grade-k component from multivector: <M>_k

    Returns grade-k blade if present, otherwise zero blade.
    """
    from morphis.geometry.model.blade import Blade

    component = M.grade_select(k)

    if component is not None:
        return component

    # Return zero blade of appropriate grade
    d = M.dim
    collection = M.collection
    shape = collection + (d,) * k if k > 0 else collection

    return Blade(data=zeros(shape), grade=k, metric=M.metric, collection=collection)


# =============================================================================
# Component Products
# =============================================================================


def scalar_product(u: Blade, v: Blade) -> NDArray:
    """
    Scalar part of geometric product: <uv>_0

    Returns scalar array with shape collection_shape.
    """
    M = geometric(u, v)
    s = M.grade_select(0)

    if s is not None:
        return s.data

    # No scalar component
    return zeros(broadcast_collection_shape(u, v))


def commutator(u: Blade, v: Blade) -> MultiVector:
    """
    Commutator product: [u, v] = (1 / 2) (uv - vu)

    Extracts antisymmetric part (odd grade differences).
    """
    uv = geometric(u, v)
    vu = geometric(v, u)

    return 0.5 * (uv - vu)


def anticommutator(u: Blade, v: Blade) -> MultiVector:
    """
    Anticommutator product: u * v = (1 / 2) (uv + vu)

    Extracts symmetric part (even grade differences).
    """
    uv = geometric(u, v)
    vu = geometric(v, u)

    return 0.5 * (uv + vu)


# =============================================================================
# Reversion
# =============================================================================


def _reverse_bl(u: Blade) -> Blade:
    """
    Reverse a blade: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k blade.

    Reverses the order of vector factors in the blade.
    """
    from morphis.geometry.model.blade import Blade

    k = u.grade
    sign = (-1) ** (k * (k - 1) // 2)

    return Blade(
        data=sign * u.data,
        grade=k,
        metric=u.metric,
        collection=u.collection,
    )


def _reverse_mv(M: MultiVector) -> MultiVector:
    """
    Reverse each component of a multivector.

    Returns MultiVector with all components reversed.
    """
    from morphis.geometry.model.multivector import MultiVector

    components = {k: _reverse_bl(blade) for k, blade in M.components.items()}

    return MultiVector(components=components, metric=M.metric)


def reverse(u: Blade | MultiVector) -> Blade | MultiVector:
    """
    Reverse: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k blade.

    For multivectors, reverses each component.

    Reverses the order of vector factors.
    """
    from morphis.geometry.model.blade import Blade

    if isinstance(u, Blade):
        return _reverse_bl(u)

    return _reverse_mv(u)


# =============================================================================
# Inverse
# =============================================================================


def _inverse_bl(u: Blade) -> Blade:
    """
    Inverse of a blade: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    """
    from morphis.geometry.model.blade import Blade

    u_rev = _reverse_bl(u)
    u_u_rev = _geometric_bl_bl(u, u_rev)

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
        metric=u.metric,
        collection=u.collection,
    )


def _inverse_mv(M: MultiVector) -> MultiVector:
    """
    Inverse of a multivector: M^(-1) = reverse(M) / (M * reverse(M))

    Requires M * reverse(M) to be invertible scalar.
    """
    M_rev = _reverse_mv(M)
    M_M_rev = _geometric_mv_mv(M, M_rev)

    # Extract scalar part
    s = M_M_rev.grade_select(0)
    if s is None:
        raise ValueError("MultiVector product with reverse is not scalar - cannot invert")

    return M_rev * (1.0 / s.data)


def inverse(u: Blade | MultiVector) -> Blade | MultiVector:
    """
    Inverse: u^(-1) = reverse(u) / (u * reverse(u))

    Requires u * reverse(u) to be nonzero scalar.
    """
    from morphis.geometry.model.blade import Blade

    if isinstance(u, Blade):
        return _inverse_bl(u)

    return _inverse_mv(u)


# =============================================================================
# Geometric Product with Mixed Types (for operators)
# =============================================================================


def geometric_bl_mv(u: Blade, M: MultiVector) -> MultiVector:
    """
    Geometric product of blade with multivector: u @ M

    Distributes over components.

    Returns MultiVector.
    """
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, M.metric)

    result_components: dict[int, Blade] = {}

    for _k, component in M.components.items():
        product = _geometric_bl_bl(u, component)

        for grade, blade in product.components.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + blade
            else:
                result_components[grade] = blade

    return MultiVector(components=result_components, metric=metric)


def geometric_mv_bl(M: MultiVector, u: Blade) -> MultiVector:
    """
    Geometric product of multivector with blade: M @ u

    Distributes over components.

    Returns MultiVector.
    """
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(M.metric, u.metric)

    result_components: dict[int, Blade] = {}

    for _k, component in M.components.items():
        product = _geometric_bl_bl(component, u)

        for grade, blade in product.components.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + blade
            else:
                result_components[grade] = blade

    return MultiVector(components=result_components, metric=metric)
