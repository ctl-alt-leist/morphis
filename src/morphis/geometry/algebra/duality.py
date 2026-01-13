"""
Geometric Algebra - Duality Operations

Complement and Hodge duality operations on Blades. These operations map
k-blades to (d-k)-blades using the Levi-Civita symbol and metric.
The metric is obtained directly from the blade's metric attribute.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from numpy import einsum

from morphis.geometry.algebra.structure import (
    INDICES,
    complement_signature,
    levi_civita,
)


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade


def right_complement(u: Blade) -> Blade:
    """
    Compute the right complement of a blade using the Levi-Civita symbol:

        comp(u)^{m_{k+1} ... m_d} = u^{m_1 ... m_k} eps_{m_1 ... m_d}

    Maps grade k blade to grade (d - k) blade, representing the orthogonal
    subspace.

    Returns Blade of grade (dim - grade) with same metric.
    """
    from morphis.geometry.model.blade import Blade

    k = u.grade
    d = u.dim
    result_grade = d - k
    eps = levi_civita(d)
    sig = complement_signature(k, d)
    result_data = einsum(sig, u.data, eps)

    return Blade(
        data=result_data,
        grade=result_grade,
        metric=u.metric,
        collection=u.collection,
    )


def left_complement(u: Blade) -> Blade:
    """
    Compute the left complement of a blade:

        lcomp(u)^{m_1 ... m_{d-k}} = eps_{m_1 ... m_d} u^{m_{d-k+1} ... m_d}

    Related to right complement by a sign factor. Maps grade k to grade (d - k).

    Returns Blade of grade (dim - grade) with same metric.
    """
    from morphis.geometry.model.blade import Blade

    k = u.grade
    d = u.dim
    result_grade = d - k
    eps = levi_civita(d)

    result_indices = INDICES[:result_grade]
    blade_indices = INDICES[result_grade : result_grade + k]
    eps_indices = result_indices + blade_indices
    sig = f"{eps_indices}, ...{blade_indices} -> ...{result_indices}"
    result_data = einsum(sig, eps, u.data)

    return Blade(
        data=result_data,
        grade=result_grade,
        metric=u.metric,
        collection=u.collection,
    )


def hodge_dual(u: Blade) -> Blade:
    """
    Compute the Hodge dual of a blade:

        *u^{m_{k+1} ... m_d} = (1/k!) u^{n_1 ... n_k} g_{n_1 m_1} ... g_{n_k m_k}
                                * eps^{m_1 ... m_d}

    Maps grade k to grade (d - k) using the metric for index lowering.
    The metric is obtained from the blade's metric attribute.

    Returns Blade of grade (dim - grade) with same metric.
    """
    from morphis.geometry.model.blade import Blade

    g = u.metric
    k = u.grade
    d = u.dim

    eps = levi_civita(d)
    result_grade = d - k

    if k == 0:
        sig = "..., " + INDICES[:d] + " -> ..." + INDICES[:d]
        result_data = einsum(sig, u.data, eps)
        return Blade(data=result_data, grade=d, metric=u.metric, collection=u.collection)

    if result_grade == 0:
        blade_indices = INDICES[:k]
        lowered_indices = INDICES[k : 2 * k]
        metric_parts = ", ".join(f"{blade_indices[m]}{lowered_indices[m]}" for m in range(k))
        sig = f"{metric_parts}, ...{blade_indices}, {lowered_indices} -> ..."
        metric_args = [g.data] * k
        result_data = einsum(sig, *metric_args, u.data, eps) / factorial(k)
        return Blade(data=result_data, grade=0, metric=u.metric, collection=u.collection)

    blade_indices = INDICES[:k]
    result_indices = INDICES[k : k + result_grade]
    lowered_indices = INDICES[k + result_grade : 2 * k + result_grade]
    metric_parts = ", ".join(f"{blade_indices[m]}{lowered_indices[m]}" for m in range(k))
    eps_sub = lowered_indices + result_indices
    sig = f"{metric_parts}, ...{blade_indices}, {eps_sub} -> ...{result_indices}"
    metric_args = [g.data] * k
    result_data = einsum(sig, *metric_args, u.data, eps) / factorial(k)

    return Blade(
        data=result_data,
        grade=result_grade,
        metric=u.metric,
        collection=u.collection,
    )
