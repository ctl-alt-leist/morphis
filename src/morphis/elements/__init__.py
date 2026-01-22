"""
Geometric Algebra - Elements

Core geometric algebra objects: Blades, MultiVectors, Frames, and Metrics.

This module provides the foundational types for geometric algebra computations:
- Blade: k-dimensional oriented subspaces
- MultiVector: general multivectors (sums of blades)
- Frame: ordered collections of vectors
- Metric: geometric context (metric tensor + signature + structure)
"""

# Base classes
# Blade and constructors
from morphis.elements.blade import (
    Blade,
    basis_blade as basis_blade,
    basis_vector as basis_vector,
    basis_vectors as basis_vectors,
    bivector_blade as bivector_blade,
    blade_from_data as blade_from_data,
    geometric_basis as geometric_basis,
    pseudoscalar as pseudoscalar,
    quadvector_blade as quadvector_blade,
    scalar_blade as scalar_blade,
    trivector_blade as trivector_blade,
    vector_blade as vector_blade,
)
from morphis.elements.elements import (
    CompositeElement as CompositeElement,
    Element as Element,
    GradedElement as GradedElement,
)

# Frame
from morphis.elements.frame import (
    Frame,
    frame_from_vectors as frame_from_vectors,
)

# Metric and context
from morphis.elements.metric import (
    PGA as PGA,
    STA as STA,
    GASignature as GASignature,
    GAStructure as GAStructure,
    Metric as Metric,
    degenerate_ns as degenerate_ns,
    euclidean as euclidean,
    euclidean_ns as euclidean_ns,
    lorentzian as lorentzian,
    lorentzian_ns as lorentzian_ns,
    metric as metric,
    pga as pga,
)

# MultiVector
from morphis.elements.multivector import (
    MultiVector,
    multivector_from_blades as multivector_from_blades,
)

# Operator (linear maps between blade spaces)
from morphis.elements.operator import Operator as Operator
from morphis.elements.protocols import Graded as Graded, Spanning as Spanning, Transformable as Transformable


# Rebuild models to resolve forward references
Blade.model_rebuild()
MultiVector.model_rebuild()
Frame.model_rebuild()
