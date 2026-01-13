"""
Geometric Algebra - Projective Operations

PGA-specific operations: point and direction embedding, geometric constructors,
distances, and incidence predicates. These operations are specific to projective
geometric algebra and use the degenerate metric diag(0, 1, 1, ..., 1).

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import (
    abs as np_abs,
    all as np_all,
    asarray,
    einsum,
    newaxis,
    sqrt,
    where,
    zeros,
)
from numpy.typing import NDArray

from morphis.geometry.algebra.norms import norm
from morphis.geometry.model.metric import Metric, pga


if TYPE_CHECKING:
    from morphis.geometry.model.blade import Blade


# =============================================================================
# Embedding
# =============================================================================


def point(x: NDArray, metric: Metric | None = None, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Embed a Euclidean point into projective space. Points have unit weight
    (e_0 component = 1):

        p = e_0 + x^1 e_1 + ... + x^d e_d

    Args:
        x: Euclidean coordinates of shape (..., d)
        metric: PGA metric (inferred from x if not provided)
        collection: Shape of the collection dimensions (inferred from x if not provided)

    Returns:
        Grade-1 blade in (d + 1)-dimensional PGA.
    """
    from morphis.geometry.model.blade import Blade

    x = asarray(x)
    d = x.shape[-1]
    dim = d + 1

    if metric is None:
        metric = pga(d)
    elif metric.dim != dim:
        raise ValueError(f"Metric dim {metric.dim} doesn't match point dim+1 {dim}")

    shape = x.shape[:-1] + (dim,)
    p = zeros(shape, dtype=x.dtype)
    p[..., 0] = 1.0
    p[..., 1:] = x

    # Infer collection from x shape if not provided
    if collection is None:
        collection = x.shape[:-1]

    return Blade(data=p, grade=1, metric=metric, collection=collection)


def direction(v: NDArray, metric: Metric | None = None, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Embed a Euclidean direction into projective space. Directions have zero
    weight (e_0 component = 0) and represent points at infinity.

    Args:
        v: Euclidean direction of shape (..., d)
        metric: PGA metric (inferred from v if not provided)
        collection: Shape of the collection dimensions (inferred from v if not provided)

    Returns:
        Grade-1 blade in (d + 1)-dimensional PGA.
    """
    from morphis.geometry.model.blade import Blade

    v = asarray(v)
    d = v.shape[-1]
    dim = d + 1

    if metric is None:
        metric = pga(d)
    elif metric.dim != dim:
        raise ValueError(f"Metric dim {metric.dim} doesn't match direction dim+1 {dim}")

    shape = v.shape[:-1] + (dim,)
    result = zeros(shape, dtype=v.dtype)
    result[..., 1:] = v

    # Infer collection from v shape if not provided
    if collection is None:
        collection = v.shape[:-1]

    return Blade(data=result, grade=1, metric=metric, collection=collection)


# =============================================================================
# Decomposition
# =============================================================================


def weight(p: Blade) -> NDArray:
    """
    Extract the weight (e_0 component) of a projective vector.

    Returns scalar array of weights.
    """
    if p.grade != 1:
        raise ValueError(f"weight() requires grade-1 blade, got grade {p.grade}")
    return p.data[..., 0]


def bulk(p: Blade) -> NDArray:
    """
    Extract the bulk (Euclidean components) of a projective vector.

    Returns array of Euclidean components with shape (*collection_shape, d).
    """
    if p.grade != 1:
        raise ValueError(f"bulk() requires grade-1 blade, got grade {p.grade}")
    return p.data[..., 1:]


def euclidean(p: Blade) -> NDArray:
    """
    Project a projective point to Euclidean coordinates by dividing bulk by
    weight. For directions (weight = 0), returns the bulk directly.

    Returns Euclidean coordinates with shape (*collection_shape, d).
    """
    w = weight(p)[..., newaxis]
    b = bulk(p)
    safe_w = where(np_abs(w) > 1e-12, w, 1.0)
    return where(np_abs(w) > 1e-12, b / safe_w, b)


# =============================================================================
# Predicates
# =============================================================================


def is_point(p: Blade) -> NDArray:
    """Check if a projective vector represents a point (nonzero weight)."""
    return np_abs(weight(p)) > 1e-12


def is_direction(p: Blade) -> NDArray:
    """Check if a projective vector represents a direction (zero weight)."""
    return np_abs(weight(p)) <= 1e-12


# =============================================================================
# Geometric Constructors
# =============================================================================


def line(p: Blade, q: Blade) -> Blade:
    """Construct a line through two points as the bivector p ^ q."""
    return p ^ q


def plane(p: Blade, q: Blade, r: Blade) -> Blade:
    """Construct a plane through three points as the trivector p ^ q ^ r."""
    return p ^ q ^ r


def plane_from_point_and_line(p: Blade, l: Blade) -> Blade:
    """Construct a plane through a point and a line as p ^ l."""
    return p ^ l


# =============================================================================
# Distances
# =============================================================================


def distance_point_to_point(p: Blade, q: Blade) -> NDArray:
    """Compute Euclidean distance between two points."""
    metric = Metric.merge(p.metric, q.metric)
    x_p = euclidean(p)
    x_q = euclidean(q)
    diff = x_q - x_p
    g_eucl = metric.data[1:, 1:]
    return sqrt(einsum("ab, ...a, ...b -> ...", g_eucl, diff, diff))


def distance_point_to_line(p: Blade, l: Blade) -> NDArray:
    """Compute distance from a point to a line as |p ^ l| / |l|."""
    Metric.merge(p.metric, l.metric)
    p_wedge_l = p ^ l
    numerator = norm(p_wedge_l)
    denominator = norm(l)
    return numerator / where(denominator > 1e-12, denominator, 1.0)


def distance_point_to_plane(p: Blade, h: Blade) -> NDArray:
    """Compute distance from a point to a plane as |p ^ h| / |h|."""
    Metric.merge(p.metric, h.metric)
    p_wedge_h = p ^ h
    numerator = norm(p_wedge_h)
    denominator = norm(h)
    return numerator / where(denominator > 1e-12, denominator, 1.0)


# =============================================================================
# Incidence Predicates
# =============================================================================


def are_collinear(p: Blade, q: Blade, r: Blade, tol: float = 1e-10) -> NDArray:
    """Check if three points are collinear: p ^ q ^ r = 0."""
    trivector = p ^ q ^ r
    return np_all(np_abs(trivector.data) < tol, axis=tuple(range(-3, 0)))


def are_coplanar(p: Blade, q: Blade, r: Blade, s: Blade, tol: float = 1e-10) -> NDArray:
    """Check if four points are coplanar: p ^ q ^ r ^ s = 0."""
    quadvector = p ^ q ^ r ^ s
    return np_all(np_abs(quadvector.data) < tol, axis=tuple(range(-4, 0)))


def point_on_line(p: Blade, l: Blade, tol: float = 1e-10) -> NDArray:
    """Check if a point lies on a line: p ^ l = 0."""
    joined = p ^ l
    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-3, 0)))


def point_on_plane(p: Blade, h: Blade, tol: float = 1e-10) -> NDArray:
    """Check if a point lies on a plane: p ^ h = 0."""
    joined = p ^ h
    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-4, 0)))


def line_in_plane(l: Blade, h: Blade, tol: float = 1e-10) -> NDArray:
    """Check if a line lies in a plane: l ^ h = 0."""
    joined = l ^ h
    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-5, 0)))
