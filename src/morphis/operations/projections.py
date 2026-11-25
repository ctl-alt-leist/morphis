"""
Geometric Algebra - Projections and Interior Products

Interior products (contractions), dot product, and projections.
The metric is obtained directly from the vector's metric attribute.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Vector naming convention: u, v, w (never a, b, c for vectors).
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from numpy import abs as np_abs, asarray, einsum, newaxis, where, zeros
from numpy.typing import NDArray

from morphis.config import TOLERANCE
from morphis.operations._helpers import broadcast_collection_shape, get_common_dim
from morphis.operations.norms import form
from morphis.operations.structure import interior_left_signature, interior_right_signature


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


# =============================================================================
# Interior Product
# =============================================================================


def _contraction_normalization(n: int) -> float:
    """
    Normalization of a contraction over n index pairs.

    The factor 1 / n! counts each set of n directions once, and the sign
    (-1)^(n (n - 1) / 2) turns the in-order pairing of the einsum signatures
    into the inside-out pairing of the geometric product.
    """
    sign = (-1) ** (n * (n - 1) // 2)

    return sign / factorial(n)


def interior_left(u: Vector, v: Vector) -> Vector:
    """
    Compute the left interior product (left contraction) of u into v:

        u _| v = <u v>_{k - j}

        (u _| v)^{n_1 ... n_{k - j}}
            = (1 / j!) u^{m_j ... m_1} v_{m_1 ... m_j}^{n_1 ... n_{k - j}}

    where indices are lowered using the metric. The indices of u pair with the
    first grade(u) indices of v from the inside out, as they do in the
    geometric product, and 1 / j! counts each set of j directions once. Hence
    e12 _| e12 = -1. Result is grade (k - j), or scalar zero if j > k.

    Both vectors must have compatible metrics (validated via Metric.merge).

    Returns Vector of grade (grade(v) - grade(u)).
    """
    from morphis.elements.metric import Metric
    from morphis.elements.vector import Vector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    get_common_dim(u, v)
    j, k = u.grade, v.grade

    # j > k: result is zero scalar
    if j > k:
        result_shape = broadcast_collection_shape(u, v)
        return Vector(data=zeros(result_shape), grade=0, metric=metric)

    # Use einsum for all cases - handles broadcasting naturally
    result_grade = k - j
    sig = interior_left_signature(j, k)
    metric_args = [g.data] * j
    result = _contraction_normalization(j) * einsum(sig, *metric_args, u.data, v.data)

    return Vector(data=result, grade=result_grade, metric=metric)


# Alias for backwards compatibility
interior = interior_left


def interior_right(u: Vector, v: Vector) -> Vector:
    """
    Compute the right interior product (right contraction) of u by v:

        u |_ v = <u v>_{j - k}

        (u |_ v)^{m_1 ... m_{j - k}}
            = (1 / k!) u^{m_1 ... m_{j - k} n_1 ... n_k} v_{n_k ... n_1}

    where indices are lowered using the metric. The indices of v pair with the
    last grade(v) indices of u from the inside out, as they do in the geometric
    product. Result is grade (j - k), or scalar zero if k > j.

    Both vectors must have compatible metrics (validated via Metric.merge).

    Returns Vector of grade (grade(u) - grade(v)).
    """
    from morphis.elements.metric import Metric
    from morphis.elements.vector import Vector

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    get_common_dim(u, v)
    j, k = u.grade, v.grade

    # k > j: result is zero scalar
    if k > j:
        result_shape = broadcast_collection_shape(u, v)
        return Vector(data=zeros(result_shape), grade=0, metric=metric)

    # Use einsum for all cases - handles broadcasting naturally
    result_grade = j - k
    sig = interior_right_signature(j, k)
    metric_args = [g.data] * k
    result = _contraction_normalization(k) * einsum(sig, *metric_args, u.data, v.data)

    return Vector(data=result, grade=result_grade, metric=metric)


# =============================================================================
# Dot Product and Projections
# =============================================================================


def dot(u: Vector, v: Vector) -> NDArray:
    """
    Compute the inner product of two grade-1 vectors: g_{mn} u^m v^n.

    Both vectors must be grade-1 and have compatible metrics.

    Returns scalar array of dot products.
    """
    from morphis.elements.metric import Metric

    # Merge metrics (raises if incompatible)
    metric = Metric.merge(u.metric, v.metric)
    g = metric

    get_common_dim(u, v)

    if u.grade != 1 or v.grade != 1:
        raise ValueError(f"dot() requires grade-1 vectors, got {u.grade} and {v.grade}")

    return einsum("mn, ...m, ...n -> ...", g.data, u.data, v.data)


def project(u: Vector, v: Vector) -> Vector:
    """
    Project u onto the subspace of the blade v:

        proj_v(u) = (u _| v) _| v^(-1)

    Projection preserves grade and is an outermorphism: proj_v(u ^ w) =
    proj_v(u) ^ proj_v(w). It is zero when grade(u) > grade(v), since a
    subspace holds no blade larger than itself. A null blade has no inverse
    and so no projection; the result there is zero.

    Both vectors must have compatible metrics.

    Returns projected vector with same grade as u.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.vector import Vector
    from morphis.operations.products import reverse

    metric = Metric.merge(u.metric, v.metric)
    d = get_common_dim(u, v)
    j, k = u.grade, v.grade

    if j > k:
        data = zeros(broadcast_collection_shape(u, v) + (d,) * j)

    else:
        # v^(-1) = reverse(v) / form(v)
        numerator = interior_left(interior_left(u, v), reverse(v))

        # A blade is null relative to its own size, whatever the metric's sign
        v_form = asarray(form(v))
        size = np_abs(v.data).max(axis=tuple(range(-k, 0))) if k > 0 else np_abs(v.data)
        is_null = np_abs(v_form) <= TOLERANCE * size**2

        for _ in range(j):
            v_form = v_form[..., newaxis]
            is_null = is_null[..., newaxis]

        data = where(is_null, 0.0, numerator.data / where(is_null, 1.0, v_form))

    return Vector(data=data, grade=j, metric=metric)


def reject(u: Vector, v: Vector) -> Vector:
    """
    Compute the rejection of u from the blade v:

        rej_v(u) = u - proj_v(u)

    Returns rejected vector with same grade as u.
    """
    return u - project(u, v)
