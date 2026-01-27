"""
Linear Algebra Module

Provides structured linear algebra utilities for geometric algebra operators.
Includes vector specifications, einsum pattern generation, and solvers.
"""

from morphis.algebra.patterns import (
    INPUT_COLLECTION as INPUT_COLLECTION,
    INPUT_GEOMETRIC as INPUT_GEOMETRIC,
    OUTPUT_COLLECTION as OUTPUT_COLLECTION,
    OUTPUT_GEOMETRIC as OUTPUT_GEOMETRIC,
    adjoint_signature as adjoint_signature,
    forward_signature as forward_signature,
    operator_shape as operator_shape,
)
from morphis.algebra.solvers import (
    structured_lstsq as structured_lstsq,
    structured_pinv as structured_pinv,
    structured_pinv_solve as structured_pinv_solve,
    structured_svd as structured_svd,
)
from morphis.algebra.specs import VectorSpec as VectorSpec, vector_spec as vector_spec
