"""
Linear Operators for Geometric Algebra

Structured linear maps between spaces of geometric objects (Blades).
Supports forward application, least-squares inverse, SVD decomposition,
and adjoint operations while maintaining tensor structure.
"""

from morphis.operations.linear.operator import LinearOperator
from morphis.operations.linear.specs import BladeSpec, blade_spec


__all__ = [
    "LinearOperator",
    "BladeSpec",
    "blade_spec",
]
