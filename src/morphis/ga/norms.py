"""
Geometric Algebra - Norms

Norm operations on Blades: squared norm, norm, and normalization.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from math import factorial

from numpy import abs as np_abs, einsum, newaxis, sqrt, where
from numpy.typing import NDArray

from morphis.ga.model import Blade, Metric, euclidean, pga
from morphis.ga.structure import norm_squared_signature


def _infer_metric(blade: Blade, explicit_metric: Metric | None) -> Metric:
    """
    Infer the appropriate metric for a blade.

    If explicit_metric is provided, use it.
    Otherwise, infer from blade context:
    - PGA context -> PGA metric (degenerate: diag(0, 1, 1, ...))
    - Other/None context -> Euclidean metric

    Args:
        blade: Blade to infer metric for
        explicit_metric: Explicitly provided metric (takes precedence)

    Returns:
        Appropriate metric for the blade
    """
    if explicit_metric is not None:
        return explicit_metric

    if blade.context is not None:
        from morphis.ga.context import Structure

        if blade.context.structure == Structure.PROJECTIVE:
            # PGA: dimension is d+1 where d is Euclidean dimension
            # pga(d) creates metric for PGA(d) which is d+1 dimensional
            return pga(blade.dim - 1)

    return euclidean(blade.dim)


def norm_squared(u: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute the squared norm of a blade:

        |u|² = (1 / k!) u^{m_1 ... m_k} u^{n_1 ... n_k} g_{m_1 n_1} ... g_{m_k n_k}

    The 1 / k! accounts for antisymmetric overcounting.

    If no metric is provided, infers from blade context:
    - PGA context -> PGA metric (degenerate)
    - Otherwise -> Euclidean metric

    Returns scalar array of squared norms with shape collection_shape.
    """
    g = _infer_metric(u, g)
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

    If no metric is provided, infers from blade context:
    - PGA context -> PGA metric (degenerate)
    - Otherwise -> Euclidean metric

    Returns scalar array of norms with shape collection_shape.
    """
    g = _infer_metric(u, g)
    return sqrt(np_abs(norm_squared(u, g)))


def normalize(u: Blade, g: Metric | None = None) -> Blade:
    """
    Normalize a blade to unit norm. Handles zero blades safely by returning
    zero.

    If no metric is provided, infers from blade context:
    - PGA context -> PGA metric (degenerate)
    - Otherwise -> Euclidean metric

    Context: Preserves input blade context.

    Returns unit blade in same direction.
    """
    g = _infer_metric(u, g)
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
