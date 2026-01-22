"""
Geometric Algebra - Products

The geometric product, wedge product, and related operations. The geometric
product unifies the dot and wedge products into a single associative operation.
For vectors u and v:

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

from morphis.operations._helpers import broadcast_collection_shape, get_broadcast_collection, get_common_dim
from morphis.operations.structure import (
    generalized_delta,
    geometric_normalization,
    geometric_signature,
    wedge_normalization,
    wedge_signature,
)


if TYPE_CHECKING:
    from morphis.elements.blade import Blade
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Wedge Product
# =============================================================================


def wedge(*blades: Blade) -> Blade:
    """
    Wedge product: u ^ v ^ ... ^ w

    Computes the exterior product of blades via antisymmetrization:

        (u ^ v)^{mn} = u^m v^n - u^n v^m

    More generally for k blades with grades (g_1, ..., g_k):

        B^{m_1 ... m_n} = outer^{a_1 ... a_n} delta^{m_1 ... m_n}_{a_1 ... a_n}

    where n = g_1 + ... + g_k and delta is the generalized Kronecker delta
    encoding antisymmetric structure.

    All blades must have compatible metrics (validated via Metric.merge).

    Returns Blade of grade sum(grades), or zero blade if sum(grades) > dim.
    """
    from morphis.elements.blade import Blade
    from morphis.elements.metric import Metric

    if not blades:
        raise ValueError("wedge() requires at least one blade")

    # Merge metrics from all blades (raises if incompatible)
    metric = Metric.merge(*(u.metric for u in blades))

    # Single blade: return copy
    if len(blades) == 1:
        u = blades[0]
        return Blade(data=u.data.copy(), grade=u.grade, metric=u.metric, collection=u.collection)

    # Dimensional and grade calculations
    d = get_common_dim(*blades)
    grades = tuple(u.grade for u in blades)
    n = sum(grades)

    # Grade exceeds dimension: result is zero
    if n > d:
        # Need explicit shape for zero blade
        collection = get_broadcast_collection(*blades)
        shape = collection + (d,) * n
        return Blade(data=zeros(shape), grade=n, metric=metric)

    # All scalars: just multiply
    if n == 0:
        result = blades[0].data
        for u in blades[1:]:
            result = result * u.data
        return Blade(data=result, grade=0, metric=metric)

    # Single einsum with delta contraction - let Blade infer collection from result
    sig = wedge_signature(grades)
    delta = generalized_delta(n, d)
    norm = wedge_normalization(grades)
    result = norm * einsum(sig, *[u.data for u in blades], delta)

    return Blade(data=result, grade=n, metric=metric)


def wedge_bl_mv(u: Blade, M: MultiVector) -> MultiVector:
    """
    Wedge product of blade with multivector: u ^ M

    Distributes over components: u ^ (A + B + ...) = (u ^ A) + (u ^ B) + ...

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    result_components: dict[int, Blade] = {}

    for _k, component in M.data.items():
        product = wedge(u, component)
        result_grade = product.grade

        if result_grade in result_components:
            result_components[result_grade] = result_components[result_grade] + product
        else:
            result_components[result_grade] = product

    return MultiVector(data=result_components, metric=Metric.merge(u.metric, M.metric))


def wedge_mv_bl(M: MultiVector, u: Blade) -> MultiVector:
    """
    Wedge product of multivector with blade: M ^ u

    Distributes over components.

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    result_components: dict[int, Blade] = {}

    for _k, component in M.data.items():
        product = wedge(component, u)
        result_grade = product.grade

        if result_grade in result_components:
            result_components[result_grade] = result_components[result_grade] + product
        else:
            result_components[result_grade] = product

    return MultiVector(data=result_components, metric=Metric.merge(M.metric, u.metric))


def wedge_mv_mv(M: MultiVector, N: MultiVector) -> MultiVector:
    """
    Wedge product of two multivectors: M ^ N

    Computes all pairwise wedge products of components and sums.

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    result_components: dict[int, Blade] = {}

    for _k1, blade1 in M.data.items():
        for _k2, blade2 in N.data.items():
            product = wedge(blade1, blade2)
            result_grade = product.grade

            if result_grade in result_components:
                result_components[result_grade] = result_components[result_grade] + product
            else:
                result_components[result_grade] = product

    return MultiVector(data=result_components, metric=Metric.merge(M.metric, N.metric))


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
    from morphis.elements.blade import Blade
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

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
            component = Blade(result_data, grade=0, metric=metric)

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

    return MultiVector(data=components, metric=metric)


# =============================================================================
# Geometric Product (MultiVector x MultiVector)
# =============================================================================


def _geometric_mv_mv(M: MultiVector, N: MultiVector) -> MultiVector:
    """
    Geometric product of two multivectors.

    Computes all pairwise geometric products of components and sums.

    Both multivectors must have compatible metrics.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(M.metric, N.metric)

    result_components: dict[int, Blade] = {}

    for _k1, u in M.data.items():
        for _k2, v in N.data.items():
            product = _geometric_bl_bl(u, v)
            for grade, component in product.data.items():
                if grade in result_components:
                    result_components[grade] = result_components[grade] + component
                else:
                    result_components[grade] = component

    return MultiVector(data=result_components, metric=metric)


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
    from morphis.elements.blade import Blade
    from morphis.elements.multivector import MultiVector

    # Both blades
    if isinstance(u, Blade) and isinstance(v, Blade):
        return _geometric_bl_bl(u, v)

    # Convert blades to multivectors if needed
    if isinstance(u, Blade):
        u = MultiVector(data={u.grade: u}, metric=u.metric)

    if isinstance(v, Blade):
        v = MultiVector(data={v.grade: v}, metric=v.metric)

    return _geometric_mv_mv(u, v)


# =============================================================================
# Grade Projection
# =============================================================================


def grade_project(M: MultiVector, k: int) -> Blade:
    """
    Extract grade-k component from multivector: <M>_k

    Returns grade-k blade if present, otherwise zero blade.
    """
    from morphis.elements.blade import Blade

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
    from morphis.elements.blade import Blade

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
    from morphis.elements.multivector import MultiVector

    components = {k: _reverse_bl(blade) for k, blade in M.data.items()}

    return MultiVector(data=components, metric=M.metric)


def reverse(u: Blade | MultiVector) -> Blade | MultiVector:
    """
    Reverse: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k blade.

    For multivectors, reverses each component.

    Reverses the order of vector factors.
    """
    from morphis.elements.blade import Blade

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
    from morphis.elements.blade import Blade

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
    from morphis.elements.blade import Blade

    if isinstance(u, Blade):
        return _inverse_bl(u)

    return _inverse_mv(u)


# =============================================================================
# Geometric Product with Mixed Types (for operators)
# =============================================================================


def geometric_bl_mv(u: Blade, M: MultiVector) -> MultiVector:
    """
    Geometric product of blade with multivector: u * M

    Distributes over components.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, M.metric)

    result_components: dict[int, Blade] = {}

    for _k, component in M.data.items():
        product = _geometric_bl_bl(u, component)

        for grade, blade in product.data.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + blade
            else:
                result_components[grade] = blade

    return MultiVector(data=result_components, metric=metric)


def geometric_mv_bl(M: MultiVector, u: Blade) -> MultiVector:
    """
    Geometric product of multivector with blade: M * u

    Distributes over components.

    Returns MultiVector.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(M.metric, u.metric)

    result_components: dict[int, Blade] = {}

    for _k, component in M.data.items():
        product = _geometric_bl_bl(component, u)

        for grade, blade in product.data.items():
            if grade in result_components:
                result_components[grade] = result_components[grade] + blade
            else:
                result_components[grade] = blade

    return MultiVector(data=result_components, metric=metric)
