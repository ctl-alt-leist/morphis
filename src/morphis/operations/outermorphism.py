"""
Geometric Algebra - Outermorphisms

An outermorphism is a linear map f: ⋀V → ⋀W that preserves the wedge product:

    f(a ∧ b) = f(a) ∧ f(b)

This preservation property means an outermorphism is completely determined by
its action on grade-1 elements (vectors). If A: V → W is the linear map on
vectors, its extension to grade-k is the k-th exterior power:

    (⋀^k A)(v₁ ∧ ... ∧ vₖ) = A(v₁) ∧ ... ∧ A(vₖ)

In components:

    (⋀^k A)(B)^{i₁...iₖ} = A^{i₁}_{m₁} ... A^{iₖ}_{mₖ} B^{m₁...mₖ}

This is k copies of A contracting with k blade indices—a natural einsum operation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from numpy import einsum

from morphis.algebra.patterns import INPUT_GEOMETRIC, OUTPUT_GEOMETRIC


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector
    from morphis.operations.operator import Operator


# =============================================================================
# Einsum Signature Generation
# =============================================================================


@lru_cache(maxsize=64)
def exterior_power_signature(k: int) -> str:
    """
    Generate einsum signature for k-th exterior power application.

    For a d×d vector map A and grade-k blade B with collection dimensions,
    computes (⋀^k A)(B) via k tensor contractions.

    Args:
        k: Grade of the blade (number of exterior power)

    Returns:
        Einsum signature string

    Examples:
        >>> exterior_power_signature(1)
        'Wa,...a->...W'

        >>> exterior_power_signature(2)
        'Wa,Xb,...ab->...WX'

        >>> exterior_power_signature(3)
        'Wa,Xb,Yc,...abc->...WXY'

    Raises:
        ValueError: If k exceeds available index pools
    """
    if k < 1:
        raise ValueError(f"Exterior power requires k >= 1, got {k}")
    if k > len(OUTPUT_GEOMETRIC):
        raise ValueError(
            f"Grade {k} exceeds output index pool size {len(OUTPUT_GEOMETRIC)}. "
            f"Maximum supported grade is {len(OUTPUT_GEOMETRIC)}."
        )
    if k > len(INPUT_GEOMETRIC):
        raise ValueError(
            f"Grade {k} exceeds input index pool size {len(INPUT_GEOMETRIC)}. "
            f"Maximum supported grade is {len(INPUT_GEOMETRIC)}."
        )

    # Build k operator subscripts: "Wa", "Xb", "Yc", ...
    op_parts = []
    for i in range(k):
        out_idx = OUTPUT_GEOMETRIC[i]
        in_idx = INPUT_GEOMETRIC[i]
        op_parts.append(f"{out_idx}{in_idx}")

    # Vector subscript: "...abc" (collection dims via ellipsis, then k geometric)
    blade_indices = INPUT_GEOMETRIC[:k]
    blade_sub = f"...{blade_indices}"

    # Result subscript: "...WXY" (collection dims via ellipsis, then k output)
    result_indices = OUTPUT_GEOMETRIC[:k]
    result_sub = f"...{result_indices}"

    return ",".join(op_parts) + f",{blade_sub}->{result_sub}"


# =============================================================================
# Exterior Power Application
# =============================================================================


def apply_exterior_power(A: "Operator", blade: "Vector", k: int) -> "Vector":
    """
    Apply k-th exterior power of a vector map to a grade-k vec.

    For a linear map A: V → W represented as a d×d matrix, and a grade-k blade
    B with components B^{m₁...mₖ}, computes:

        (⋀^k A)(B)^{i₁...iₖ} = A^{i₁}_{m₁} ... A^{iₖ}_{mₖ} B^{m₁...mₖ}

    This is k copies of A contracting with the k geometric indices of B.

    Args:
        A: Operator with grade-1 → grade-1 mapping (the vector map)
        blade: Grade-k blade to transform
        k: Grade of the blade

    Returns:
        Transformed grade-k blade

    Raises:
        ValueError: If A is not a grade-1 → grade-1 operator
        ValueError: If blade grade doesn't match k
    """
    from morphis.elements.vector import Vector

    if A.input_spec.grade != 1 or A.output_spec.grade != 1:
        raise ValueError(
            f"Exterior power requires grade-1 → grade-1 operator, "
            f"got grade-{A.input_spec.grade} → grade-{A.output_spec.grade}"
        )

    if blade.grade != k:
        raise ValueError(f"Vector grade {blade.grade} doesn't match k={k}")

    if k == 0:
        # Scalars are invariant under outermorphisms
        return blade.copy()

    if k == 1:
        # Direct application for vectors
        return A.apply(blade)

    # Extract the d×d vector map matrix
    vector_map = A.vector_map

    # Build einsum signature and apply k copies of the vector map
    sig = exterior_power_signature(k)
    result_data = einsum(sig, *([vector_map] * k), blade.data)

    return Vector(
        data=result_data,
        grade=k,
        metric=A.metric,
    )


# =============================================================================
# Outermorphism Application to MultiVector
# =============================================================================


def apply_outermorphism(A: "Operator", M: "MultiVector") -> "MultiVector":
    """
    Apply an outermorphism to a multivector.

    An outermorphism is a linear map that preserves the wedge product structure.
    It is completely determined by its action on grade-1 (vectors), which then
    extends to all grades via exterior powers.

    For each grade-k component of M, applies the k-th exterior power ⋀^k A.

    Args:
        A: Operator with grade-1 → grade-1 mapping (defines the outermorphism)
        M: MultiVector to transform

    Returns:
        Transformed MultiVector with all grades mapped

    Raises:
        TypeError: If A is not a grade-1 → grade-1 operator

    Examples:
        >>> # Rotation matrix as outermorphism
        >>> R = Operator(rotation_matrix, ...)  # grade-1 → grade-1
        >>> M_rotated = apply_outermorphism(R, M)
    """
    from morphis.elements.multivector import MultiVector

    if A.input_spec.grade != 1 or A.output_spec.grade != 1:
        raise TypeError(
            f"Outermorphism requires grade-1 → grade-1 operator. "
            f"This operator maps grade-{A.input_spec.grade} → grade-{A.output_spec.grade}. "
            f"Use Operator.apply() for single-grade linear maps."
        )

    # Check for collection dimensions - not yet supported
    if A.input_spec.collection != 0 or A.output_spec.collection != 0:
        raise NotImplementedError(
            "Outermorphism with collection dimensions not yet implemented. "
            f"Operator has input_collection={A.input_spec.collection}, "
            f"output_collection={A.output_spec.collection}."
        )

    result: dict[int, Vector] = {}

    for k, blade in M.data.items():
        result[k] = apply_exterior_power(A, blade, k)

    return MultiVector(data=result, metric=A.metric)
