"""
Geometric Algebra - Projections and Interior Products

Interior products (contractions), dot product, and projections.
The metric is obtained directly from the blade's metric attribute.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import einsum, newaxis, where, zeros
from numpy.typing import NDArray

from morphis.operations._helpers import broadcast_collection_shape, get_common_dim
from morphis.operations.norms import norm_squared
from morphis.operations.structure import interior_left_signature, interior_right_signature


if TYPE_CHECKING:
    from morphis.elements.blade import Blade


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
    from morphis.elements.blade import Blade
    from morphis.elements.metric import Metric

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
    from morphis.elements.blade import Blade
    from morphis.elements.metric import Metric

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
# Dot Product and Projections
# =============================================================================


def dot(u: Blade, v: Blade) -> NDArray:
    """
    Compute the inner product of two vectors: g_{mn} u^m v^n.

    Both blades must be grade-1 and have compatible metrics.

    Returns scalar array of dot products.
    """
    from morphis.elements.metric import Metric

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
    from morphis.elements.blade import Blade

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
