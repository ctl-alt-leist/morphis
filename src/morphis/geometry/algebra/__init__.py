"""
Geometric Algebra - Algebra Operations

This module provides the algebraic operations for geometric algebra:
- Wedge product (exterior product)
- Interior products (left and right contractions)
- Geometric product
- Duality operations (complements, Hodge dual)
- Norms and normalization
- Factorization
- Projective operations (PGA)
"""

# Duality
from morphis.geometry.algebra.duality import (
    hodge_dual,
    left_complement,
    right_complement,
)

# Factorization
from morphis.geometry.algebra.factorization import (
    factor,
    spanning_vectors,
)

# Geometric product
from morphis.geometry.algebra.geometric import (
    anticommutator,
    commutator,
    geometric,
    geometric_bl_mv,
    geometric_mv_bl,
    grade_project,
    inverse,
    reverse,
    scalar_product,
)

# Norms
from morphis.geometry.algebra.norms import (
    norm,
    norm_squared,
    normalize,
)

# Operations
from morphis.geometry.algebra.operations import (
    dot,
    interior,
    interior_left,
    interior_right,
    join,
    meet,
    project,
    reject,
    wedge,
    wedge_bl_mv,
    wedge_mv_bl,
    wedge_mv_mv,
)
from morphis.geometry.algebra.projective import (
    are_collinear,
    are_coplanar,
    bulk,
    direction,
    distance_point_to_line,
    distance_point_to_plane,
    distance_point_to_point,
    euclidean,
    is_direction,
    is_point,
    line,
    line_in_plane,
    plane,
    plane_from_point_and_line,
    point,
    point_on_line,
    point_on_plane,
    weight,
)
from morphis.geometry.algebra.structure import (
    INDICES,
    antisymmetric_symbol,
    antisymmetrize,
    complement_signature,
    generalized_delta,
    geometric_normalization,
    geometric_signature,
    interior_left_signature,
    interior_right_signature,
    interior_signature,
    levi_civita,
    norm_squared_signature,
    permutation_sign,
    wedge_normalization,
    wedge_signature,
)


__all__ = [
    # Structure
    "INDICES",
    "antisymmetric_symbol",
    "antisymmetrize",
    "complement_signature",
    "generalized_delta",
    "geometric_normalization",
    "geometric_signature",
    "interior_left_signature",
    "interior_right_signature",
    "interior_signature",
    "levi_civita",
    "norm_squared_signature",
    "permutation_sign",
    "wedge_normalization",
    "wedge_signature",
    # Norms
    "norm",
    "norm_squared",
    "normalize",
    # Duality
    "hodge_dual",
    "left_complement",
    "right_complement",
    # Operations
    "dot",
    "interior",
    "interior_left",
    "interior_right",
    "join",
    "meet",
    "project",
    "reject",
    "wedge",
    "wedge_bl_mv",
    "wedge_mv_bl",
    "wedge_mv_mv",
    # Geometric product
    "anticommutator",
    "commutator",
    "geometric",
    "geometric_bl_mv",
    "geometric_mv_bl",
    "grade_project",
    "inverse",
    "reverse",
    "scalar_product",
    # Factorization
    "factor",
    "spanning_vectors",
    # Projective (PGA)
    "point",
    "direction",
    "weight",
    "bulk",
    "euclidean",
    "is_point",
    "is_direction",
    "line",
    "plane",
    "plane_from_point_and_line",
    "distance_point_to_point",
    "distance_point_to_line",
    "distance_point_to_plane",
    "are_collinear",
    "are_coplanar",
    "point_on_line",
    "point_on_plane",
    "line_in_plane",
]
