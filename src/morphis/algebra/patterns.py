"""
Linear Algebra - Einsum Pattern Generation

Generates einsum signatures for linear operator operations. Uses disjoint index
pools to avoid collisions between input and output indices.

Index naming convention:
- OUTPUT_LOT: "KLMN" (up to 4 output lot dims)
- INPUT_LOT: "nopq" (up to 4 input lot dims)
- OUTPUT_GEOMETRIC: "WXYZ" (up to grade-4 output blades)
- INPUT_GEOMETRIC: "abcd" (up to grade-4 input blades)

Storage conventions (lot-first, matching Vector layout):
- Operator: (*out_lot, *in_lot, *out_geo, *in_geo)
- Vector: (*lot, *geo)
"""

from functools import lru_cache

from morphis.algebra.specs import VectorSpec


# Index pools (disjoint to avoid collisions)
OUTPUT_LOT = "KLMN"
INPUT_LOT = "nopq"
OUTPUT_GEOMETRIC = "WXYZ"
INPUT_GEOMETRIC = "abcd"

# Backwards compatibility aliases
OUTPUT_COLLECTION = OUTPUT_LOT
INPUT_COLLECTION = INPUT_LOT


@lru_cache(maxsize=128)
def forward_signature(input_spec: VectorSpec, output_spec: VectorSpec) -> str:
    """
    Generate einsum signature for forward operator application: y = L * x

    Operator data has shape: (*out_lot, *in_lot, *out_geo, *in_geo)
    Input Vector has shape: (*in_lot, *in_geo)
    Output Vector has shape: (*out_lot, *out_geo)

    Args:
        input_spec: Specification of input Vector
        output_spec: Specification of output Vector

    Returns:
        Einsum signature string, e.g., "KnWX,n->KWX" for scalar->bivector

    Examples:
        >>> # Scalar currents (N,) to bivector fields (M, 3, 3)
        >>> sig = forward_signature(
        ...     VectorSpec(grade=0, lot=(1,), dim=3),
        ...     VectorSpec(grade=2, lot=(1,), dim=3),
        ... )
        >>> sig
        'KnWX,n->KWX'

        >>> # Vector (N, 3) to bivector (M, 3, 3)
        >>> sig = forward_signature(
        ...     VectorSpec(grade=1, lot=(1,), dim=3),
        ...     VectorSpec(grade=2, lot=(1,), dim=3),
        ... )
        >>> sig
        'KnWXa,na->KWX'
    """
    _validate_spec_limits(input_spec, output_spec)

    # Output lot indices
    out_lot = OUTPUT_LOT[: output_spec.collection]

    # Input lot indices (contracted)
    in_lot = INPUT_LOT[: input_spec.collection]

    # Output geometric indices
    out_geo = OUTPUT_GEOMETRIC[: output_spec.grade]

    # Input geometric indices (contracted)
    in_geo = INPUT_GEOMETRIC[: input_spec.grade]

    # Build operator signature: out_lot + in_lot + out_geo + in_geo (lot-first)
    op_indices = out_lot + in_lot + out_geo + in_geo

    # Build input signature: in_lot + in_geo (Vector storage order)
    input_indices = in_lot + in_geo

    # Build output signature: out_lot + out_geo (Vector storage order)
    output_indices = out_lot + out_geo

    return f"{op_indices},{input_indices}->{output_indices}"


@lru_cache(maxsize=128)
def adjoint_signature(input_spec: VectorSpec, output_spec: VectorSpec) -> str:
    """
    Generate einsum signature for adjoint operator application: x = L^H * y

    The adjoint contracts over output indices (what were previously the result).

    Args:
        input_spec: Specification of original input (becomes adjoint output)
        output_spec: Specification of original output (becomes adjoint input)

    Returns:
        Einsum signature string for adjoint application

    Examples:
        >>> # Adjoint of scalar->bivector: bivector->scalar
        >>> sig = adjoint_signature(
        ...     VectorSpec(grade=0, lot=(1,), dim=3),
        ...     VectorSpec(grade=2, lot=(1,), dim=3),
        ... )
        >>> sig
        'KnWX,KWX->n'
    """
    _validate_spec_limits(input_spec, output_spec)

    # Same index allocation as forward
    out_lot = OUTPUT_LOT[: output_spec.collection]
    in_lot = INPUT_LOT[: input_spec.collection]
    out_geo = OUTPUT_GEOMETRIC[: output_spec.grade]
    in_geo = INPUT_GEOMETRIC[: input_spec.grade]

    # Operator indices: out_lot + in_lot + out_geo + in_geo (lot-first)
    op_indices = out_lot + in_lot + out_geo + in_geo

    # Adjoint input (original output vec): out_lot + out_geo
    adjoint_input = out_lot + out_geo

    # Adjoint output (original input space): in_lot + in_geo
    adjoint_output = in_lot + in_geo

    return f"{op_indices},{adjoint_input}->{adjoint_output}"


def operator_shape(
    input_spec: VectorSpec,
    output_spec: VectorSpec,
    input_lot: tuple[int, ...] | None = None,
    output_lot: tuple[int, ...] | None = None,
    *,
    input_collection: tuple[int, ...] | None = None,
    output_collection: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """
    Compute the expected shape of operator data given specs and lot shapes.

    Args:
        input_spec: Specification of input Vector
        output_spec: Specification of output Vector
        input_lot: Shape of input lot dimensions
        output_lot: Shape of output lot dimensions
        input_collection: DEPRECATED alias for input_lot
        output_collection: DEPRECATED alias for output_lot

    Returns:
        Operator data shape: (*out_lot, *in_lot, *out_geo, *in_geo)

    Examples:
        >>> # Scalar (N=5) to bivector (M=10) in 3D
        >>> shape = operator_shape(
        ...     VectorSpec(grade=0, lot=(1,), dim=3),
        ...     VectorSpec(grade=2, lot=(1,), dim=3),
        ...     input_lot=(5,),
        ...     output_lot=(10,),
        ... )
        >>> shape
        (10, 5, 3, 3)
    """
    # Handle deprecated aliases
    if input_lot is None and input_collection is not None:
        input_lot = input_collection
    if output_lot is None and output_collection is not None:
        output_lot = output_collection

    if input_lot is None or output_lot is None:
        raise ValueError("input_lot and output_lot are required")

    if len(input_lot) != input_spec.collection:
        raise ValueError(f"input_lot has {len(input_lot)} dims, but input_spec expects {input_spec.collection}")
    if len(output_lot) != output_spec.collection:
        raise ValueError(f"output_lot has {len(output_lot)} dims, but output_spec expects {output_spec.collection}")

    return output_lot + input_lot + output_spec.geo + input_spec.geo


def _validate_spec_limits(input_spec: VectorSpec, output_spec: VectorSpec) -> None:
    """Validate that specs are within index pool limits."""
    if input_spec.grade > len(INPUT_GEOMETRIC):
        raise ValueError(f"Input grade {input_spec.grade} exceeds index pool limit {len(INPUT_GEOMETRIC)}")
    if output_spec.grade > len(OUTPUT_GEOMETRIC):
        raise ValueError(f"Output grade {output_spec.grade} exceeds index pool limit {len(OUTPUT_GEOMETRIC)}")
    if input_spec.collection > len(INPUT_LOT):
        raise ValueError(f"Input lot dims {input_spec.collection} exceeds limit {len(INPUT_LOT)}")
    if output_spec.collection > len(OUTPUT_LOT):
        raise ValueError(f"Output lot dims {output_spec.collection} exceeds limit {len(OUTPUT_LOT)}")
