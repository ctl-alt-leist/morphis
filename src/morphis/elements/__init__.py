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
from morphis.elements.elements import CompositeElement, Element, GradedElement

# Frame
from morphis.elements.frame import (
    Frame,
    frame_from_vectors,
)

# Metric and context
from morphis.elements.metric import (
    PGA,
    STA,
    GASignature,
    GAStructure,
    Metric,
    degenerate_ns,
    euclidean,
    euclidean_ns,
    lorentzian,
    lorentzian_ns,
    pga,
)

# MultiVector
from morphis.elements.multivector import (
    MultiVector,
    multivector_from_blades,
)
from morphis.elements.protocols import Graded, Spanning, Transformable


# Rebuild models to resolve forward references
Blade.model_rebuild()
MultiVector.model_rebuild()
Frame.model_rebuild()

__all__ = [
    # Base classes
    "Element",
    "GradedElement",
    "CompositeElement",
    # Protocols
    "Graded",
    "Spanning",
    "Transformable",
    # Metric
    "GASignature",
    "GAStructure",
    "Metric",
    "euclidean",
    "pga",
    "lorentzian",
    "euclidean_ns",
    "lorentzian_ns",
    "degenerate_ns",
    "PGA",
    "STA",
    # Blade
    "Blade",
    "scalar_blade",
    "vector_blade",
    "bivector_blade",
    "trivector_blade",
    "quadvector_blade",
    "blade_from_data",
    "basis_vector",
    "basis_vectors",
    "basis_blade",
    "pseudoscalar",
    # MultiVector
    "MultiVector",
    "multivector_from_blades",
    # Frame
    "Frame",
    "frame_from_vectors",
]
