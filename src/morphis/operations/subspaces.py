"""
Geometric Algebra - Subspace Operations

Join and meet operations for computing unions and intersections of subspaces.

Blade naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from morphis.operations.duality import right_complement
from morphis.operations.products import wedge


if TYPE_CHECKING:
    from morphis.elements.blade import Blade


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

        u v v = dual(dual(u) ^ dual(v))

    where dual denotes the right complement.

    Returns Blade representing the intersection.
    """
    u_comp = right_complement(u)
    v_comp = right_complement(v)
    joined = wedge(u_comp, v_comp)

    return right_complement(joined)
