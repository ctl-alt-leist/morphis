"""
Linear Algebra Module

Provides structured linear algebra utilities for geometric algebra operators.
Includes blade specifications, einsum pattern generation, and solvers.
"""

from morphis.algebra.patterns import (
    INPUT_COLLECTION,
    INPUT_GEOMETRIC,
    OUTPUT_COLLECTION,
    OUTPUT_GEOMETRIC,
    adjoint_signature,
    forward_signature,
    operator_shape,
)
from morphis.algebra.solvers import (
    structured_lstsq,
    structured_pinv,
    structured_pinv_solve,
    structured_svd,
)
from morphis.algebra.specs import BladeSpec, blade_spec


__all__ = [
    # Specifications
    "BladeSpec",
    "blade_spec",
    # Patterns
    "forward_signature",
    "adjoint_signature",
    "operator_shape",
    "OUTPUT_GEOMETRIC",
    "OUTPUT_COLLECTION",
    "INPUT_COLLECTION",
    "INPUT_GEOMETRIC",
    # Solvers
    "structured_lstsq",
    "structured_svd",
    "structured_pinv",
    "structured_pinv_solve",
]
