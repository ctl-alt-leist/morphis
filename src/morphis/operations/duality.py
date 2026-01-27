"""
Geometric Algebra - Duality Operations

Complement and Hodge duality operations on Vectors. These operations map
k-vectors to (d-k)-vectors using the Levi-Civita symbol and metric.
The metric is obtained directly from the vector's metric attribute.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Vector naming convention: u, v, w (never a, b, c for vectors).
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from numpy import einsum

from morphis.operations.structure import (
    INDICES,
    complement_signature,
    levi_civita,
)


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


def right_complement(u: Vector) -> Vector:
    """
    Compute the right complement of a vector using the Levi-Civita symbol:

        comp(u)^{m_{k+1} ... m_d} = u^{m_1 ... m_k} eps_{m_1 ... m_d}

    Maps grade k vector to grade (d - k) vector, representing the orthogonal
    subspace.

    Returns Vector of grade (dim - grade) with same metric.
    """
    from morphis.elements.vector import Vector

    k = u.grade
    d = u.dim
    result_grade = d - k
    eps = levi_civita(d)
    sig = complement_signature(k, d)
    result_data = einsum(sig, u.data, eps)

    return Vector(
        data=result_data,
        grade=result_grade,
        metric=u.metric,
        collection=u.collection,
    )


def left_complement(u: Vector) -> Vector:
    """
    Compute the left complement of a vector:

        lcomp(u)^{m_1 ... m_{d-k}} = eps_{m_1 ... m_d} u^{m_{d-k+1} ... m_d}

    Related to right complement by a sign factor. Maps grade k to grade (d - k).

    Returns Vector of grade (dim - grade) with same metric.
    """
    from morphis.elements.vector import Vector

    k = u.grade
    d = u.dim
    result_grade = d - k
    eps = levi_civita(d)

    result_indices = INDICES[:result_grade]
    vector_indices = INDICES[result_grade : result_grade + k]
    eps_indices = result_indices + vector_indices
    sig = f"{eps_indices}, ...{vector_indices} -> ...{result_indices}"
    result_data = einsum(sig, eps, u.data)

    return Vector(
        data=result_data,
        grade=result_grade,
        metric=u.metric,
        collection=u.collection,
    )


def hodge_dual(u: Vector) -> Vector:
    """
    Compute the Hodge dual of a vector:

        *u^{m_{k+1} ... m_d} = (1/k!) u^{n_1 ... n_k} g_{n_1 m_1} ... g_{n_k m_k}
                                * eps^{m_1 ... m_d}

    Maps grade k to grade (d - k) using the metric for index lowering.
    The metric is obtained from the vector's metric attribute.

    Returns Vector of grade (dim - grade) with same metric.
    """
    from morphis.elements.vector import Vector

    g = u.metric
    k = u.grade
    d = u.dim

    eps = levi_civita(d)
    result_grade = d - k

    if k == 0:
        sig = "..., " + INDICES[:d] + " -> ..." + INDICES[:d]
        result_data = einsum(sig, u.data, eps)
        return Vector(data=result_data, grade=d, metric=u.metric, collection=u.collection)

    if result_grade == 0:
        vector_indices = INDICES[:k]
        lowered_indices = INDICES[k : 2 * k]
        metric_parts = ", ".join(f"{vector_indices[m]}{lowered_indices[m]}" for m in range(k))
        sig = f"{metric_parts}, ...{vector_indices}, {lowered_indices} -> ..."
        metric_args = [g.data] * k
        result_data = einsum(sig, *metric_args, u.data, eps) / factorial(k)
        return Vector(data=result_data, grade=0, metric=u.metric, collection=u.collection)

    vector_indices = INDICES[:k]
    result_indices = INDICES[k : k + result_grade]
    lowered_indices = INDICES[k + result_grade : 2 * k + result_grade]
    metric_parts = ", ".join(f"{vector_indices[m]}{lowered_indices[m]}" for m in range(k))
    eps_sub = lowered_indices + result_indices
    sig = f"{metric_parts}, ...{vector_indices}, {eps_sub} -> ...{result_indices}"
    metric_args = [g.data] * k
    result_data = einsum(sig, *metric_args, u.data, eps) / factorial(k)

    return Vector(
        data=result_data,
        grade=result_grade,
        metric=u.metric,
        collection=u.collection,
    )
