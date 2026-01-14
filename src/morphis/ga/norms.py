"""
Geometric Algebra - Norms

Norm operations on Blades: squared norm, norm, and normalization.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from math import factorial

from numpy import abs as np_abs, einsum, newaxis, sqrt, where
from numpy.typing import NDArray

from morphis.ga.model import Blade, Metric, euclidean
from morphis.ga.structure import norm_squared_signature


def norm_squared(u: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute the squared norm of a blade:

        |u|² = (1 / k!) u^{m_1 ... m_k} u^{n_1 ... n_k} g_{m_1 n_1} ... g_{m_k n_k}

    The 1 / k! accounts for antisymmetric overcounting.

    Returns scalar array of squared norms with shape collection_shape.
    """
    g = euclidean(u.dim) if g is None else g
    k = u.grade
    d = u.dim

    if d != g.dim:
        raise ValueError(f"Blade dim {d} != metric dim {g.dim}")

    if k == 0:
        return u.data * u.data

    sig = norm_squared_signature(k)
    metric_args = [g.data] * k
    return einsum(sig, *metric_args, u.data, u.data) / factorial(k)


def norm(u: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute the norm of a blade as sqrt of absolute value of norm squared.

    Returns scalar array of norms with shape collection_shape.
    """
    g = euclidean(u.dim) if g is None else g
    return sqrt(np_abs(norm_squared(u, g)))


def normalize(u: Blade, g: Metric | None = None) -> Blade:
    """
    Normalize a blade to unit norm. Handles zero blades safely by returning
    zero.

    Context: Preserves input blade context.

    Returns unit blade in same direction.
    """
    g = euclidean(u.dim) if g is None else g
    n = norm(u, g)
    n_expanded = n
    for _ in range(u.grade):
        n_expanded = n_expanded[..., newaxis]

    safe_norm = where(n_expanded > 1e-12, n_expanded, 1.0)
    result_data = u.data / safe_norm

    return Blade(
        data=result_data,
        grade=u.grade,
        dim=u.dim,
        cdim=u.cdim,
        context=u.context,
    )
