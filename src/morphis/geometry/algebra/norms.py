"""
Geometric Algebra - Norms

Norm operations on Blades: squared norm, norm, and normalization.
The metric is obtained directly from the blade's metric attribute.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from numpy import abs as np_abs, einsum, newaxis, sqrt, where
from numpy.typing import NDArray

from morphis.geometry.algebra.structure import norm_squared_signature


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade


def norm_squared(u: Blade) -> NDArray:
    """
    Compute the squared norm of a blade:

        |u|^2 = (1 / k!) u^{m_1 ... m_k} u^{n_1 ... n_k} g_{m_1 n_1} ... g_{m_k n_k}

    The 1 / k! accounts for antisymmetric overcounting.
    The metric is obtained from the blade's metric attribute.

    Returns scalar array of squared norms with shape collection_shape.
    """
    k = u.grade
    g = u.metric

    if k == 0:
        return u.data * u.data

    sig = norm_squared_signature(k)
    metric_args = [g.data] * k
    return einsum(sig, *metric_args, u.data, u.data) / factorial(k)


def norm(u: Blade) -> NDArray:
    """
    Compute the norm of a blade as sqrt of absolute value of norm squared.

    The metric is obtained from the blade's metric attribute.

    Returns scalar array of norms with shape collection_shape.
    """
    return sqrt(np_abs(norm_squared(u)))


def normalize(u: Blade) -> Blade:
    """
    Normalize a blade to unit norm.

    Handles zero blades safely by returning zero.
    The metric is obtained from the blade's metric attribute.

    Returns unit blade in same direction with same metric.
    """
    from morphis.geometry.model.blade import Blade

    n = norm(u)
    n_expanded = n
    for _ in range(u.grade):
        n_expanded = n_expanded[..., newaxis]

    safe_norm = where(n_expanded > 1e-12, n_expanded, 1.0)
    result_data = u.data / safe_norm

    return Blade(
        data=result_data,
        grade=u.grade,
        metric=u.metric,
        collection=u.collection,
    )
