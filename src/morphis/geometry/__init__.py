"""
Morphis Geometry - Geometric Algebra

A complete geometric algebra implementation with:
- Blades and MultiVectors with required Metric context
- Wedge, interior, and geometric products
- Rotors and translators for transformations
- Projective (PGA) operations for Euclidean geometry

Key Types:
    Metric: Complete geometric context (tensor + signature + structure)
    Blade: k-dimensional oriented subspace
    MultiVector: General element (sum of blades)
    Frame: Collection of vectors

Quick Start:
    >>> from morphis.geometry import euclidean, pga, Blade, point, rotor
    >>>
    >>> # Euclidean VGA
    >>> m = euclidean(3)
    >>> v = Blade([1, 0, 0], grade=1, metric=m)
    >>>
    >>> # Projective GA (3D Euclidean via 4D PGA)
    >>> m = pga(3)
    >>> p = point([1, 2, 3])  # Creates PGA point
"""

# =============================================================================
# Model Types
# =============================================================================

# =============================================================================
# Algebra Operations
# =============================================================================
from morphis.geometry.algebra.duality import (
    hodge_dual,
    left_complement,
    right_complement,
)
from morphis.geometry.algebra.factorization import (
    factor,
    spanning_vectors,
)
from morphis.geometry.algebra.geometric import (
    anticommutator,
    commutator,
    geometric,
    grade_project,
    inverse,
    reverse,
    scalar_product,
)
from morphis.geometry.algebra.norms import norm, norm_squared, normalize
from morphis.geometry.algebra.operations import (
    dot,
    interior_left,
    interior_right,
    join,
    meet,
    project,
    reject,
    wedge,
)

# =============================================================================
# Projective Operations (PGA)
# =============================================================================
from morphis.geometry.algebra.projective import (
    are_collinear,
    are_coplanar,
    bulk,
    direction,
    distance_point_to_line,
    distance_point_to_plane,
    distance_point_to_point,
    euclidean as euclidean_coords,
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
from morphis.geometry.model.blade import (
    Blade,
    basis_blade,
    basis_vector,
    basis_vectors,
    bivector_blade,
    blade_from_data,
    pseudoscalar,
    quadvector_blade,
    scalar_blade,
    trivector_blade,
    vector_blade,
)
from morphis.geometry.model.frame import Frame, frame_from_vectors
from morphis.geometry.model.metric import (
    GASignature,
    GAStructure,
    Metric,
    euclidean,
    lorentzian,
    pga,
)
from morphis.geometry.model.multivector import MultiVector, multivector_from_blades

# =============================================================================
# Transformations
# =============================================================================
from morphis.geometry.transforms import (
    rotate,
    rotation_about_point,
    rotor,
    screw_motion,
    transform,
    translate,
    translator,
)


__all__ = [
    # Metric and context
    "Metric",
    "GASignature",
    "GAStructure",
    "euclidean",
    "pga",
    "lorentzian",
    # Core types
    "Blade",
    "MultiVector",
    "Frame",
    # Blade constructors
    "scalar_blade",
    "vector_blade",
    "bivector_blade",
    "trivector_blade",
    "quadvector_blade",
    "blade_from_data",
    "multivector_from_blades",
    "frame_from_vectors",
    # Basis constructors
    "basis_vector",
    "basis_vectors",
    "basis_blade",
    "pseudoscalar",
    # Factorization
    "factor",
    "spanning_vectors",
    # Algebra operations
    "wedge",
    "interior_left",
    "interior_right",
    "dot",
    "project",
    "reject",
    "join",
    "meet",
    "geometric",
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
    # Transformations
    "rotor",
    "translator",
    "rotation_about_point",
    "screw_motion",
    "rotate",
    "translate",
    "transform",
    # Projective operations
    "point",
    "direction",
    "weight",
    "bulk",
    "euclidean_coords",
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
