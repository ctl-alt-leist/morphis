"""
Geometric Algebra - Projective Operations

PGA-specific operations: point and direction embedding, geometric constructors,
distances, and incidence predicates. These operations are specific to projective
geometric algebra and use the degenerate metric diag(0, 1, 1, ..., 1).

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

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

from morphis.ga.context import degenerate
from morphis.ga.model import Blade, Metric, pga
from morphis.ga.norms import norm
from morphis.ga.operations import wedge


# =============================================================================
# Embedding
# =============================================================================


def point(x: NDArray, cdim: int = 0) -> Blade:
    """
    Embed a Euclidean point into projective space. Points have unit weight
    (e_0 component = 1):

        p = e_0 + x^1 e_1 + ... + x^d e_d

    Context: Sets degenerate.projective (PGA) automatically.

    Returns grade-1 blade in (d + 1)-dimensional PGA.
    """
    x = asarray(x)
    d = x.shape[-1]
    shape = x.shape[:-1] + (d + 1,)
    p = zeros(shape, dtype=x.dtype)
    p[..., 0] = 1.0
    p[..., 1:] = x

    return Blade(data=p, grade=1, dim=d + 1, cdim=cdim, context=degenerate.projective)


def direction(v: NDArray, cdim: int = 0) -> Blade:
    """
    Embed a Euclidean direction into projective space. Directions have zero
    weight (e_0 component = 0) and represent points at infinity.

    Context: Sets degenerate.projective (PGA) automatically.

    Returns grade-1 blade in (d + 1)-dimensional PGA.
    """
    v = asarray(v)
    d = v.shape[-1]
    shape = v.shape[:-1] + (d + 1,)
    result = zeros(shape, dtype=v.dtype)
    result[..., 1:] = v

    return Blade(data=result, grade=1, dim=d + 1, cdim=cdim, context=degenerate.projective)


# =============================================================================
# Decomposition
# =============================================================================


def weight(p: Blade) -> NDArray:
    """
    Extract the weight (e_0 component) of a projective vector. Points have
    unit weight; directions have zero weight.

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
    """
    Check if a projective vector represents a point (nonzero weight).

    Returns boolean array.
    """
    return np_abs(weight(p)) > 1e-12


def is_direction(p: Blade) -> NDArray:
    """
    Check if a projective vector represents a direction (zero weight).

    Returns boolean array.
    """
    return np_abs(weight(p)) <= 1e-12


# =============================================================================
# Geometric Constructors
# =============================================================================


def line(p: Blade, q: Blade) -> Blade:
    """
    Construct a line through two points as the bivector p ∧ q.

    Returns grade-2 blade representing the line.
    """
    return wedge(p, q)


def plane(p: Blade, q: Blade, r: Blade) -> Blade:
    """
    Construct a plane through three points as the trivector p ∧ q ∧ r.

    Returns grade-3 blade representing the plane.
    """
    return wedge(wedge(p, q), r)


def plane_from_point_and_line(p: Blade, l: Blade) -> Blade:
    """
    Construct a plane through a point and a line as p ∧ l.

    Returns grade-3 blade representing the plane.
    """
    return wedge(p, l)


# =============================================================================
# Distances
# =============================================================================


def distance_point_to_point(p: Blade, q: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute Euclidean distance between two points.

    Returns scalar array of distances.
    """
    g = pga(p.dim - 1) if g is None else g
    x_p = euclidean(p)
    x_q = euclidean(q)
    diff = x_q - x_p
    g_eucl = g.data[1:, 1:]

    return sqrt(einsum("ab, ...a, ...b -> ...", g_eucl, diff, diff))


def distance_point_to_line(p: Blade, l: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute distance from a point to a line as |p ∧ l| / |l|.

    Returns scalar array of distances.
    """
    g = pga(p.dim - 1) if g is None else g
    p_wedge_l = wedge(p, l)
    numerator = norm(p_wedge_l, g)
    denominator = norm(l, g)

    return numerator / where(denominator > 1e-12, denominator, 1.0)


def distance_point_to_plane(p: Blade, h: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute distance from a point to a plane (hyperplane) as |p ∧ h| / |h|.

    Returns scalar array of distances.
    """
    g = pga(p.dim - 1) if g is None else g
    p_wedge_h = wedge(p, h)
    numerator = norm(p_wedge_h, g)
    denominator = norm(h, g)

    return numerator / where(denominator > 1e-12, denominator, 1.0)


# =============================================================================
# Incidence Predicates
# =============================================================================


def are_collinear(p: Blade, q: Blade, r: Blade, tol: float = 1e-10) -> NDArray:
    """
    Check if three points are collinear: p ∧ q ∧ r = 0.

    Returns boolean array.
    """
    trivector = wedge(wedge(p, q), r)

    return np_all(np_abs(trivector.data) < tol, axis=tuple(range(-3, 0)))


def are_coplanar(p: Blade, q: Blade, r: Blade, s: Blade, tol: float = 1e-10) -> NDArray:
    """
    Check if four points are coplanar: p ∧ q ∧ r ∧ s = 0.

    Returns boolean array.
    """
    quadvector = wedge(wedge(wedge(p, q), r), s)

    return np_all(np_abs(quadvector.data) < tol, axis=tuple(range(-4, 0)))


def point_on_line(p: Blade, l: Blade, tol: float = 1e-10) -> NDArray:
    """
    Check if a point lies on a line: p ∧ l = 0.

    Returns boolean array.
    """
    joined = wedge(p, l)

    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-3, 0)))


def point_on_plane(p: Blade, h: Blade, tol: float = 1e-10) -> NDArray:
    """
    Check if a point lies on a plane: p ∧ h = 0.

    Returns boolean array.
    """
    joined = wedge(p, h)

    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-4, 0)))


def line_in_plane(l: Blade, h: Blade, tol: float = 1e-10) -> NDArray:
    """
    Check if a line lies in a plane: l ∧ h = 0.

    Returns boolean array.
    """
    joined = wedge(l, h)

    return np_all(np_abs(joined.data) < tol, axis=tuple(range(-5, 0)))
