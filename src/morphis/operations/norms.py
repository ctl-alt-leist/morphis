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

from numpy import abs as np_abs, conj, einsum, newaxis, sqrt, where
from numpy.typing import NDArray

from morphis.operations.structure import norm_squared_signature


if TYPE_CHECKING:
    from morphis.elements.blade import Blade


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
    from morphis.elements.blade import Blade

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


def conjugate(u: Blade) -> Blade:
    """
    Return blade with complex-conjugated coefficients.

    For real blades, returns a copy (conjugation is identity on reals).
    For complex blades, applies np.conj to all coefficients.

    This is the coefficient conjugation, not a GA operation. The complex
    numbers represent temporal phasors, not geometric structure.

    Returns Blade with conjugated data.
    """
    from morphis.elements.blade import Blade

    return Blade(
        data=conj(u.data),
        grade=u.grade,
        metric=u.metric,
        collection=u.collection,
    )


def hermitian_norm_squared(u: Blade) -> NDArray:
    """
    Compute Hermitian (sesquilinear) squared norm:

        |u|^2_H = (1 / k!) conj(u^{m_1 ... m_k}) u^{n_1 ... n_k} g_{m_1 n_1} ... g_{m_k n_k}

    This is the physical magnitude squared, always real for real metrics.
    For real blades, equivalent to norm_squared.
    For complex blades (phasors), gives the squared RMS amplitude.

    Use this for physical quantities. Use norm_squared for algebraic
    (bilinear) inner product computations.

    Returns real scalar array of squared norms with shape collection_shape.
    """
    k = u.grade
    g = u.metric

    if k == 0:
        return (conj(u.data) * u.data).real

    sig = norm_squared_signature(k)
    metric_args = [g.data] * k
    result = einsum(sig, *metric_args, conj(u.data), u.data) / factorial(k)
    return result.real


def hermitian_norm(u: Blade) -> NDArray:
    """
    Compute Hermitian norm: sqrt of hermitian_norm_squared.

    Always returns real non-negative values for positive-definite metrics.
    For real blades, equivalent to norm.
    For complex blades (phasors), gives the RMS amplitude.

    Returns real scalar array of norms with shape collection_shape.
    """
    return sqrt(hermitian_norm_squared(u))
