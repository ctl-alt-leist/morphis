"""
Geometric Algebra - Operations

Algebraic operations on blades and multivectors: products, norms, duality,
projections, subspaces, and factorization.
"""

# Products (geometric, wedge, reverse, inverse)
# Duality
from morphis.operations.duality import (
    hodge_dual as hodge_dual,
    left_complement as left_complement,
    right_complement as right_complement,
)

# Factorization
from morphis.operations.factorization import (
    factor as factor,
    spanning_vectors as spanning_vectors,
)

# Norms
from morphis.operations.norms import (
    conjugate as conjugate,
    hermitian_norm as hermitian_norm,
    hermitian_norm_squared as hermitian_norm_squared,
    norm as norm,
    norm_squared as norm_squared,
    normalize as normalize,
)
from morphis.operations.products import (
    anticommutator as anticommutator,
    commutator as commutator,
    geometric as geometric,
    geometric_bl_mv as geometric_bl_mv,
    geometric_mv_bl as geometric_mv_bl,
    grade_project as grade_project,
    inverse as inverse,
    reverse as reverse,
    scalar_product as scalar_product,
    wedge as wedge,
    wedge_bl_mv as wedge_bl_mv,
    wedge_mv_bl as wedge_mv_bl,
    wedge_mv_mv as wedge_mv_mv,
)

# Projections and interior products
from morphis.operations.projections import (
    dot as dot,
    interior as interior,
    interior_left as interior_left,
    interior_right as interior_right,
    project as project,
    reject as reject,
)

# Structure constants (for advanced users)
from morphis.operations.structure import (
    INDICES as INDICES,
    antisymmetric_symbol as antisymmetric_symbol,
    antisymmetrize as antisymmetrize,
    complement_signature as complement_signature,
    generalized_delta as generalized_delta,
    geometric_normalization as geometric_normalization,
    geometric_signature as geometric_signature,
    interior_left_signature as interior_left_signature,
    interior_right_signature as interior_right_signature,
    interior_signature as interior_signature,
    levi_civita as levi_civita,
    norm_squared_signature as norm_squared_signature,
    permutation_sign as permutation_sign,
    wedge_normalization as wedge_normalization,
    wedge_signature as wedge_signature,
)

# Subspaces
from morphis.operations.subspaces import (
    join as join,
    meet as meet,
)
