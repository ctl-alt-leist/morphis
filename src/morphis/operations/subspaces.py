"""
Geometric Algebra - Subspace Operations

Join and meet operations for computing unions and intersections of subspaces.

These are aliases for wedge and antiwedge products, using subspace terminology:
- join(u, v) = wedge(u, v) = u ∧ v — smallest subspace containing both
- meet(u, v) = antiwedge(u, v) = u ∨ v — largest subspace contained in both

Vector naming convention: u, v, w (never a, b, c for blades).
"""

from __future__ import annotations

from morphis.operations.products import antiwedge, wedge


# Join is the wedge product (union of subspaces)
join = wedge

# Meet is the antiwedge/regressive product (intersection of subspaces)
meet = antiwedge
