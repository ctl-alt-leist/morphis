"""
Linear Algebra - Einsum Pattern Generation

Generates einsum signatures for linear operator operations. Uses disjoint index
pools to avoid collisions between input and output indices.

Index naming convention:
- OUTPUT_GEOMETRIC: "WXYZ" (up to grade-4 output blades)
- OUTPUT_COLLECTION: "KLMN" (up to 4 output collection dims)
- INPUT_COLLECTION: "nopq" (up to 4 input collection dims)
- INPUT_GEOMETRIC: "abcd" (up to grade-4 input blades)

Operator storage convention: (*output_geometric, *output_collection, *input_collection, *input_geometric)
Vector storage convention: (*collection, *geometric)
"""

from functools import lru_cache

from morphis.algebra.specs import VectorSpec


# Index pools (disjoint to avoid collisions)
OUTPUT_GEOMETRIC = "WXYZ"
OUTPUT_COLLECTION = "KLMN"
INPUT_COLLECTION = "nopq"
INPUT_GEOMETRIC = "abcd"


@lru_cache(maxsize=128)
def forward_signature(input_spec: VectorSpec, output_spec: VectorSpec) -> str:
    """
    Generate einsum signature for forward operator application: y = L * x

    Operator data has shape: (*output_geometric, *output_collection, *input_collection, *input_geometric)
    Input blade has shape: (*input_collection, *input_geometric)
    Output blade has shape: (*output_collection, *output_geometric)

    Args:
        input_spec: Specification of input blade
        output_spec: Specification of output blade

    Returns:
        Einsum signature string, e.g., "WXKn,n->KWX" for scalar->bivector

    Examples:
        >>> # Scalar currents (N,) to bivector fields (M, 3, 3)
        >>> sig = forward_signature(
        ...     VectorSpec(grade=0, collection=1, dim=3),
        ...     VectorSpec(grade=2, collection=1, dim=3),
        ... )
        >>> sig
        'WXKn,n->KWX'

        >>> # Vector (N, 3) to bivector (M, 3, 3)
        >>> sig = forward_signature(
        ...     VectorSpec(grade=1, collection=1, dim=3),
        ...     VectorSpec(grade=2, collection=1, dim=3),
        ... )
        >>> sig
        'WXKna,na->KWX'
    """
    _validate_spec_limits(input_spec, output_spec)

    # Output geometric indices (stored first in operator)
    out_geo = OUTPUT_GEOMETRIC[: output_spec.grade]

    # Output collection indices
    out_coll = OUTPUT_COLLECTION[: output_spec.collection]

    # Input collection indices (contracted)
    in_coll = INPUT_COLLECTION[: input_spec.collection]

    # Input geometric indices (contracted)
    in_geo = INPUT_GEOMETRIC[: input_spec.grade]

    # Build operator signature: out_geo + out_coll + in_coll + in_geo
    op_indices = out_geo + out_coll + in_coll + in_geo

    # Build input signature: in_coll + in_geo (blade storage order)
    input_indices = in_coll + in_geo

    # Build output signature: out_coll + out_geo (blade storage order)
    output_indices = out_coll + out_geo

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
        ...     VectorSpec(grade=0, collection=1, dim=3),
        ...     VectorSpec(grade=2, collection=1, dim=3),
        ... )
        >>> sig
        'WXKn,KWX->n'
    """
    _validate_spec_limits(input_spec, output_spec)

    # Same index allocation as forward
    out_geo = OUTPUT_GEOMETRIC[: output_spec.grade]
    out_coll = OUTPUT_COLLECTION[: output_spec.collection]
    in_coll = INPUT_COLLECTION[: input_spec.collection]
    in_geo = INPUT_GEOMETRIC[: input_spec.grade]

    # Operator indices unchanged: out_geo + out_coll + in_coll + in_geo
    op_indices = out_geo + out_coll + in_coll + in_geo

    # Adjoint input (original output vec): out_coll + out_geo
    adjoint_input = out_coll + out_geo

    # Adjoint output (original input space): in_coll + in_geo
    adjoint_output = in_coll + in_geo

    return f"{op_indices},{adjoint_input}->{adjoint_output}"


def operator_shape(
    input_spec: VectorSpec,
    output_spec: VectorSpec,
    input_collection: tuple[int, ...],
    output_collection: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Compute the expected shape of operator data given specs and collection shapes.

    Args:
        input_spec: Specification of input blade
        output_spec: Specification of output blade
        input_collection: Shape of input collection dimensions
        output_collection: Shape of output collection dimensions

    Returns:
        Operator data shape: (*output_geometric, *output_collection, *input_collection, *input_geometric)

    Examples:
        >>> # Scalar (N=5) to bivector (M=10) in 3D
        >>> shape = operator_shape(
        ...     VectorSpec(grade=0, collection=1, dim=3),
        ...     VectorSpec(grade=2, collection=1, dim=3),
        ...     input_collection=(5,),
        ...     output_collection=(10,),
        ... )
        >>> shape
        (3, 3, 10, 5)
    """
    if len(input_collection) != input_spec.collection:
        raise ValueError(
            f"input_collection has {len(input_collection)} dims, but input_spec expects {input_spec.collection}"
        )
    if len(output_collection) != output_spec.collection:
        raise ValueError(
            f"output_collection has {len(output_collection)} dims, but output_spec expects {output_spec.collection}"
        )

    return output_spec.geometric_shape + output_collection + input_collection + input_spec.geometric_shape


def _validate_spec_limits(input_spec: VectorSpec, output_spec: VectorSpec) -> None:
    """Validate that specs are within index pool limits."""
    if input_spec.grade > len(INPUT_GEOMETRIC):
        raise ValueError(f"Input grade {input_spec.grade} exceeds index pool limit {len(INPUT_GEOMETRIC)}")
    if output_spec.grade > len(OUTPUT_GEOMETRIC):
        raise ValueError(f"Output grade {output_spec.grade} exceeds index pool limit {len(OUTPUT_GEOMETRIC)}")
    if input_spec.collection > len(INPUT_COLLECTION):
        raise ValueError(f"Input collection {input_spec.collection} exceeds limit {len(INPUT_COLLECTION)}")
    if output_spec.collection > len(OUTPUT_COLLECTION):
        raise ValueError(f"Output collection {output_spec.collection} exceeds limit {len(OUTPUT_COLLECTION)}")
