"""
Geometric Algebra - Operations

Algebraic operations on Blades: wedge product, interior product, join, meet,
dot product, and projections. These operations work with any metric and
support collection dimensions via einsum broadcasting.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
Blade naming convention: u, v, w (never a, b, c for blades).
"""

from numpy import einsum, newaxis, where, zeros
from numpy.typing import NDArray

from morphis.ga.context import GeometricContext
from morphis.ga.duality import right_complement
from morphis.ga.model import Blade, Metric, euclidean
from morphis.ga.norms import norm_squared
from morphis.ga.structure import (
    generalized_delta,
    interior_signature,
    wedge_normalization,
    wedge_signature,
)
from morphis.ga.utils import (
    broadcast_collection_shape,
    get_common_cdim,
    get_common_dim,
)


# =============================================================================
# Wedge Product
# =============================================================================


def wedge(*blades: Blade) -> Blade:
    """
    Wedge product: u ∧ v ∧ ... ∧ w

    Computes the exterior product of blades via antisymmetrization:

        (u ∧ v)^{mn} = u^m v^n - u^n v^m

    More generally for k blades with grades (g_1, ..., g_k):

        B^{m_1 ... m_n} = outer^{a_1 ... a_n} δ^{m_1 ... m_n}_{a_1 ... a_n}

    where n = g_1 + ... + g_k and δ is the generalized Kronecker delta
    encoding antisymmetric structure.

    The result satisfies:
    - Anticommutativity: u ∧ v = -v ∧ u (for odd grade blades)
    - Associativity: (u ∧ v) ∧ w = u ∧ (v ∧ w)
    - Unit norm: |e_i ∧ e_j| = 1 for orthonormal basis vectors

    Context: Preserves context if all inputs match, otherwise None.

    Returns Blade of grade sum(grades), or zero blade if sum(grades) > dim.
    """
    if not blades:
        raise ValueError("wedge() requires at least one blade")

    # Merge contexts from all blades
    merged_context = GeometricContext.merge(*[u.context for u in blades])

    # Single blade: return copy
    if len(blades) == 1:
        u = blades[0]
        return Blade(data=u.data.copy(), grade=u.grade, dim=u.dim, cdim=u.cdim, context=u.context)

    # Dimensional and grade calculations - important for signature
    d = get_common_dim(*blades)
    cdim = get_common_cdim(*blades)
    grades = tuple(u.grade for u in blades)
    n = sum(grades)

    # Grade exceeds dimension: result is zero
    if n > d:
        shape = tuple(1 for _ in range(cdim)) + (d,) * n
        return Blade(data=zeros(shape), grade=n, dim=d, cdim=cdim, context=merged_context)

    # All scalars: just multiply
    if n == 0:
        result = blades[0].data
        for u in blades[1:]:
            result = result * u.data
        return Blade(data=result, grade=0, dim=d, cdim=cdim, context=merged_context)

    # Single einsum with delta contraction
    sig = wedge_signature(grades)
    delta = generalized_delta(n, d)
    norm = wedge_normalization(grades)
    result = norm * einsum(sig, *[u.data for u in blades], delta)

    return Blade(data=result, grade=n, dim=d, cdim=cdim, context=merged_context)


# =============================================================================
# Interior Product
# =============================================================================


def interior(u: Blade, v: Blade, g: Metric | None = None) -> Blade:
    """
    Compute the interior product (left contraction) of u into v:

        (u ⌋ v)^{n_1 ... n_{k - j}}
            = u^{m_1 ... m_j} v_{m_1 ... m_j}^{n_1 ... n_{k - j}}

    where indices are lowered using the metric g. Contracts all indices of u
    with the first grade(u) indices of v. Result is grade (k - j), or zero
    blade if j > k.

    Context: Preserves context if both inputs match, otherwise None.

    Returns Blade of grade (grade(v) - grade(u)).
    """
    g = euclidean(u.dim) if g is None else g
    d = get_common_dim(u, v)

    if d != g.dim:
        raise ValueError(f"Blade dim {d} != metric dim {g.dim}")

    j, k = u.grade, v.grade
    result_cdim = get_common_cdim(u, v)
    merged_context = GeometricContext.merge(u.context, v.context)

    if j > k:
        result_shape = broadcast_collection_shape(u, v)
        return Blade(data=zeros(result_shape), grade=0, dim=d, cdim=result_cdim, context=merged_context)

    if j == 0:
        if k == 0:
            return Blade(data=u.data * v.data, grade=0, dim=d, cdim=result_cdim, context=merged_context)

        result_grade = k
        result = u.data[..., newaxis] * v.data
        while result.ndim < result_cdim + result_grade:
            result = result[..., newaxis]

        return Blade(
            data=result,
            grade=result_grade,
            dim=d,
            cdim=result_cdim,
            context=merged_context,
        )

    result_grade = k - j
    sig = interior_signature(j, k)
    metric_args = [g.data] * j
    result = einsum(sig, *metric_args, u.data, v.data)

    return Blade(
        data=result,
        grade=result_grade,
        dim=d,
        cdim=result_cdim,
        context=merged_context,
    )


# =============================================================================
# Join and Meet
# =============================================================================


def join(u: Blade, v: Blade) -> Blade:
    """
    Compute the join of two blades: the smallest subspace containing both.
    Algebraically, join is the wedge product.

    Returns Blade representing the joined subspace.
    """
    return wedge(u, v)


def meet(u: Blade, v: Blade) -> Blade:
    """
    Compute the meet (intersection) of two blades: the largest subspace
    contained in both. Computed via duality:

        u ∨ v = (ū ∧ v̄)̄

    where bar denotes the right complement.

    Returns Blade representing the intersection.
    """
    u_comp = right_complement(u)
    v_comp = right_complement(v)
    joined = wedge(u_comp, v_comp)

    return right_complement(joined)


# =============================================================================
# Dot Product and Projections
# =============================================================================


def dot(u: Blade, v: Blade, g: Metric | None = None) -> NDArray:
    """
    Compute the inner product of two vectors: g_{mn} u^m v^n.

    Returns scalar array of dot products.
    """
    d = get_common_dim(u, v)
    g = euclidean(d) if g is None else g

    if u.grade != 1 or v.grade != 1:
        raise ValueError(f"dot() requires grade-1 blades, got {u.grade} and {v.grade}")
    if d != g.dim:
        raise ValueError(f"Blade dim {d} != metric dim {g.dim}")

    return einsum("mn, ...m, ...n -> ...", g.data, u.data, v.data)


def project(u: Blade, v: Blade, g: Metric | None = None) -> Blade:
    """
    Project blade u onto blade v:

        proj_v(u) = (u ⌋ v) ⌋ v / |v|²

    Context: Preserves context if both inputs match, otherwise None.

    Returns projected blade with same grade as u.
    """
    d = get_common_dim(u, v)
    g = euclidean(d) if g is None else g

    contraction = interior(u, v, g)
    result = interior(contraction, v, g)
    v_norm_sq = norm_squared(v, g)

    n_expanded = v_norm_sq
    for _ in range(result.grade):
        n_expanded = n_expanded[..., newaxis]

    safe_norm = where(n_expanded > 1e-12, n_expanded, 1.0)

    return Blade(
        data=result.data / safe_norm,
        grade=result.grade,
        dim=result.dim,
        cdim=result.cdim,
        context=result.context,
    )


def reject(u: Blade, v: Blade, g: Metric | None = None) -> Blade:
    """
    Compute the rejection of blade u from blade v: the component of u
    orthogonal to v.

    Returns rejected blade with same grade as u.
    """
    d = get_common_dim(u, v)
    g = euclidean(d) if g is None else g
    return u - project(u, v, g)
