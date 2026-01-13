"""
Geometric Algebra - Model Types

This module provides the core data structures for geometric algebra:
- Metric: Complete geometric context (tensor + signature + structure)
- Blade: k-dimensional oriented subspace
- MultiVector: Sum of blades of different grades
- Frame: Ordered collection of vectors
"""

# Base model
from morphis.geometry.model.base import GAModel

# Core types
from morphis.geometry.model.blade import (
    Blade,
    # Basis constructors
    basis_blade,
    basis_vector,
    basis_vectors,
    bivector_blade,
    blade_from_data,
    pseudoscalar,
    quadvector_blade,
    # Constructor functions
    scalar_blade,
    trivector_blade,
    vector_blade,
)
from morphis.geometry.model.frame import (
    Frame,
    frame_from_vectors,
)

# Metric and context
from morphis.geometry.model.metric import (
    # Aliases
    PGA,
    STA,
    GASignature,
    GAStructure,
    Metric,
    degenerate_ns,
    # Factory functions
    euclidean,
    # Namespace access (for euclidean.flat(dim) style)
    euclidean_ns,
    lorentzian,
    lorentzian_ns,
    pga,
)
from morphis.geometry.model.multivector import (
    MultiVector,
    multivector_from_blades,
)

# Protocols
from morphis.geometry.model.protocols import GAObject, Transformable


# Rebuild models to resolve forward references (must be after all imports)
Blade.model_rebuild()
MultiVector.model_rebuild()


__all__ = [
    # Base
    "GAModel",
    # Protocols
    "GAObject",
    "Transformable",
    # Metric types
    "GASignature",
    "GAStructure",
    "Metric",
    # Metric factories
    "euclidean",
    "pga",
    "lorentzian",
    # Metric namespaces
    "euclidean_ns",
    "lorentzian_ns",
    "degenerate_ns",
    # Metric aliases
    "PGA",
    "STA",
    # Core types
    "Blade",
    "MultiVector",
    "Frame",
    # Constructor functions
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
]
