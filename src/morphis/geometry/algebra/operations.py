"""
Geometric Algebra - Operations

Algebraic operations on Blades: wedge product, interior product, join, meet,
dot product, and projections. Metrics are obtained from blade attributes and
validated for compatibility using Metric.merge().

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import einsum, newaxis, where, zeros
from numpy.typing import NDArray

from morphis.geometry.algebra.duality import right_complement
from morphis.geometry.algebra.norms import norm_squared
from morphis.geometry.algebra.structure import (
    generalized_delta,
    interior_left_signature,
    interior_right_signature,
    wedge_normalization,
    wedge_signature,
)
from morphis.geometry.utils import (
    broadcast_collection_shape,
    get_broadcast_collection,
    get_common_dim,
)


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.multivector import MultiVector


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
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.metric import Metric

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


# =============================================================================
# Interior Product
# =============================================================================


def interior_left(u: Blade, v: Blade) -> Blade:
    """
    Compute the left interior product (left contraction) of u into v:

        (u _| v)^{n_1 ... n_{k - j}}
            = u^{m_1 ... m_j} v_{m_1 ... m_j}^{n_1 ... n_{k - j}}

    where indices are lowered using the metric. Contracts all indices of u
    with the first grade(u) indices of v. Result is grade (k - j), or zero
    blade if j > k.

    Both blades must have compatible metrics (validated via Metric.merge).

    Returns Blade of grade (grade(v) - grade(u)).
    """
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.metric import Metric

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    get_common_dim(u, v)
    j, k = u.grade, v.grade

    # j > k: result is zero scalar
    if j > k:
        result_shape = broadcast_collection_shape(u, v)
        return Blade(data=zeros(result_shape), grade=0, metric=metric)

    # Use einsum for all cases - handles broadcasting naturally
    result_grade = k - j
    sig = interior_left_signature(j, k)
    metric_args = [g.data] * j
    result = einsum(sig, *metric_args, u.data, v.data)

    return Blade(data=result, grade=result_grade, metric=metric)


# Alias for backwards compatibility
interior = interior_left


def interior_right(u: Blade, v: Blade) -> Blade:
    """
    Compute the right interior product (right contraction) of u by v:

        (u |_ v)^{m_1 ... m_{j - k}}
            = u^{m_1 ... m_{j - k} n_1 ... n_k} v_{n_1 ... n_k}

    where indices are lowered using the metric. Contracts all indices of v
    with the last grade(v) indices of u. Result is grade (j - k), or zero
    blade if k > j.

    Both blades must have compatible metrics (validated via Metric.merge).

    Returns Blade of grade (grade(u) - grade(v)).
    """
    from morphis.geometry.model.blade import Blade
    from morphis.geometry.model.metric import Metric

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    get_common_dim(u, v)
    j, k = u.grade, v.grade

    # k > j: result is zero scalar
    if k > j:
        result_shape = broadcast_collection_shape(u, v)
        return Blade(data=zeros(result_shape), grade=0, metric=metric)

    # Use einsum for all cases - handles broadcasting naturally
    result_grade = j - k
    sig = interior_right_signature(j, k)
    metric_args = [g.data] * k
    result = einsum(sig, *metric_args, u.data, v.data)

    return Blade(data=result, grade=result_grade, metric=metric)


# =============================================================================
# Join and Meet
# =============================================================================


def join(u: Blade, v: Blade) -> Blade:
    """
    Compute the join of two blades: the smallest subspace containing both.
    Algebraically, join is the wedge product.

    Returns Blade representing the joined subspace.
    """
    return wedge(u, v)


def meet(u: Blade, v: Blade) -> Blade:
    """
    Compute the meet (intersection) of two blades: the largest subspace
    contained in both. Computed via duality:

        u v v = dual(dual(u) ^ dual(v))

    where dual denotes the right complement.

    Returns Blade representing the intersection.
    """
    u_comp = right_complement(u)
    v_comp = right_complement(v)
    joined = wedge(u_comp, v_comp)

    return right_complement(joined)


# =============================================================================
# Dot Product and Projections
# =============================================================================


def dot(u: Blade, v: Blade) -> NDArray:
    """
    Compute the inner product of two vectors: g_{mn} u^m v^n.

    Both blades must be grade-1 and have compatible metrics.

    Returns scalar array of dot products.
    """
    from morphis.geometry.model.metric import Metric

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    get_common_dim(u, v)

    if u.grade != 1 or v.grade != 1:
        raise ValueError(f"dot() requires grade-1 blades, got {u.grade} and {v.grade}")

    return einsum("mn, ...m, ...n -> ...", g.data, u.data, v.data)


def project(u: Blade, v: Blade) -> Blade:
    """
    Project blade u onto blade v:

        proj_v(u) = (u _| v) _| v / |v|^2

    Both blades must have compatible metrics.

    Returns projected blade with same grade as u.
    """
    from morphis.geometry.model.blade import Blade

    contraction = interior_left(u, v)
    result = interior_left(contraction, v)
    v_norm_sq = norm_squared(v)

    n_expanded = v_norm_sq
    for _ in range(result.grade):
        n_expanded = n_expanded[..., newaxis]

    safe_norm = where(n_expanded > 1e-12, n_expanded, 1.0)

    return Blade(
        data=result.data / safe_norm,
        grade=result.grade,
        metric=result.metric,
        collection=result.collection,
    )


def reject(u: Blade, v: Blade) -> Blade:
    """
    Compute the rejection of blade u from blade v: the component of u
    orthogonal to v.

    Returns rejected blade with same grade as u.
    """
    return u - project(u, v)


# =============================================================================
# Wedge Product with MultiVectors
# =============================================================================


def wedge_bl_mv(u: Blade, M: MultiVector) -> MultiVector:
    """
    Wedge product of blade with multivector: u ^ M

    Distributes over components: u ^ (A + B + ...) = (u ^ A) + (u ^ B) + ...

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    result_components: dict[int, Blade] = {}

    for _k, component in M.components.items():
        product = wedge(u, component)
        result_grade = product.grade

        if result_grade in result_components:
            result_components[result_grade] = result_components[result_grade] + product
        else:
            result_components[result_grade] = product

    return MultiVector(components=result_components, metric=Metric.merge(u.metric, M.metric))


def wedge_mv_bl(M: MultiVector, u: Blade) -> MultiVector:
    """
    Wedge product of multivector with blade: M ^ u

    Distributes over components.

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    result_components: dict[int, Blade] = {}

    for _k, component in M.components.items():
        product = wedge(component, u)
        result_grade = product.grade

        if result_grade in result_components:
            result_components[result_grade] = result_components[result_grade] + product
        else:
            result_components[result_grade] = product

    return MultiVector(components=result_components, metric=Metric.merge(M.metric, u.metric))


def wedge_mv_mv(M: MultiVector, N: MultiVector) -> MultiVector:
    """
    Wedge product of two multivectors: M ^ N

    Computes all pairwise wedge products of components and sums.

    All components must have compatible metrics.

    Returns MultiVector.
    """
    from morphis.geometry.model.metric import Metric
    from morphis.geometry.model.multivector import MultiVector

    result_components: dict[int, Blade] = {}

    for _k1, blade1 in M.components.items():
        for _k2, blade2 in N.components.items():
            product = wedge(blade1, blade2)
            result_grade = product.grade

            if result_grade in result_components:
                result_components[result_grade] = result_components[result_grade] + product
            else:
                result_components[result_grade] = product

    return MultiVector(components=result_components, metric=Metric.merge(M.metric, N.metric))
