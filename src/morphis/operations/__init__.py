"""
Geometric Algebra - Operations

Algebraic operations on blades and multivectors: products, norms, duality,
projections, subspaces, and factorization.
"""

# Products (geometric, wedge, reverse, inverse)
# Duality
from morphis.operations.duality import (
    hodge_dual,
    left_complement,
    right_complement,
)

# Factorization
from morphis.operations.factorization import (
    factor,
    spanning_vectors,
)

# Norms
from morphis.operations.norms import (
    norm,
    norm_squared,
    normalize,
)
from morphis.operations.products import (
    anticommutator,
    commutator,
    geometric,
    geometric_bl_mv,
    geometric_mv_bl,
    grade_project,
    inverse,
    reverse,
    scalar_product,
    wedge,
    wedge_bl_mv,
    wedge_mv_bl,
    wedge_mv_mv,
)

# Projections and interior products
from morphis.operations.projections import (
    dot,
    interior,
    interior_left,
    interior_right,
    project,
    reject,
)

# Structure constants (for advanced users)
from morphis.operations.structure import (
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

# Subspaces
from morphis.operations.subspaces import (
    join,
    meet,
)


__all__ = [
    # Products
    "geometric",
    "geometric_bl_mv",
    "geometric_mv_bl",
    "wedge",
    "wedge_bl_mv",
    "wedge_mv_bl",
    "wedge_mv_mv",
    "reverse",
    "inverse",
    "grade_project",
    "scalar_product",
    "commutator",
    "anticommutator",
    # Norms
    "norm",
    "norm_squared",
    "normalize",
    # Duality
    "hodge_dual",
    "left_complement",
    "right_complement",
    # Projections
    "dot",
    "interior",
    "interior_left",
    "interior_right",
    "project",
    "reject",
    # Subspaces
    "join",
    "meet",
    # Factorization
    "factor",
    "spanning_vectors",
    # Structure constants
    "INDICES",
    "permutation_sign",
    "antisymmetrize",
    "antisymmetric_symbol",
    "levi_civita",
    "generalized_delta",
    "wedge_signature",
    "wedge_normalization",
    "interior_signature",
    "interior_left_signature",
    "interior_right_signature",
    "complement_signature",
    "norm_squared_signature",
    "geometric_signature",
    "geometric_normalization",
]
