"""
Linear Algebra - Tensor Contraction

Provides two contraction APIs for Morphis tensors:
1. Index notation: G["mnab"] * q["n"] - bracket syntax with IndexedTensor
2. Einsum-style: contract("mnab, n -> mab", G, q) - explicit signature
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import einsum


if TYPE_CHECKING:
    from morphis.elements.vector import Vector
    from morphis.operations.operator import Operator


# =============================================================================
# IndexedTensor - Bracket Syntax API
# =============================================================================


class IndexedTensor:
    """
    Lightweight wrapper that pairs a tensor with index labels for contraction.

    This class enables einsum-style syntax:
        G["mnab"] * q["n"]  # contracts on index 'n'

    The wrapper holds a reference (not a copy) to the underlying tensor,
    making indexing O(1). Computation only happens when two IndexedTensors
    are multiplied.

    Attributes:
        tensor: The underlying Vector or Operator (reference, not copy)
        indices: String of index labels (e.g., "mnab")
        output_geo_indices: Indices that represent output geometric dimensions.
            These determine the result grade when present in contraction output.

    Examples:
        >>> G = Operator(...)  # lot=(M, N), grade=2 output
        >>> q = Vector(...)  # lot=(N,), grade=0
        >>> b = G["mnab"] * q["n"]  # contracts on 'n', result has indices "mab"
    """

    __slots__ = ("tensor", "indices", "output_geo_indices")

    def __init__(self, tensor: "Vector | Operator", indices: str, output_geo_indices: str = ""):
        """
        Create an indexed tensor wrapper.

        Args:
            tensor: The underlying Vector or Operator
            indices: String of index labels, one per axis of tensor.data
            output_geo_indices: Subset of indices representing output geometric
                dimensions. If empty, will be inferred during contraction (legacy).
        """
        self.tensor = tensor
        self.indices = indices
        self.output_geo_indices = output_geo_indices

        # Validate index count matches tensor dimensions
        expected_ndim = tensor.data.ndim
        if len(indices) != expected_ndim:
            raise ValueError(
                f"Index string '{indices}' has {len(indices)} indices, but tensor has {expected_ndim} dimensions"
            )

    def __mul__(self, other: "IndexedTensor") -> "Vector":
        """
        Contract two indexed tensors on matching indices.

        Args:
            other: Another IndexedTensor or LotIndexed to contract with

        Returns:
            Vector with the contracted result
        """
        from morphis.elements.lot_indexed import LotIndexed

        if isinstance(other, LotIndexed):
            # Convert LotIndexed to IndexedTensor by adding geo indices
            # All of a Vector's geometric indices are "output geometric"
            n_geo = other.vector.grade
            geo_labels = "".join(chr(ord("A") + i) for i in range(n_geo))
            other = IndexedTensor(
                other.vector,
                other.indices + geo_labels,
                output_geo_indices=geo_labels,
            )

        if not isinstance(other, IndexedTensor):
            return NotImplemented

        return _contract_indexed(self, other)

    def __rmul__(self, other: "IndexedTensor") -> "Vector":
        """Right multiplication for contraction."""
        if not isinstance(other, IndexedTensor):
            return NotImplemented

        return _contract_indexed(other, self)

    def __repr__(self) -> str:
        tensor_type = type(self.tensor).__name__
        return f"IndexedTensor({tensor_type}, indices='{self.indices}')"


def _contract_indexed(*indexed_tensors: IndexedTensor) -> "Vector":
    """
    Contract multiple IndexedTensor objects.

    Internal function that performs the actual contraction for bracket syntax.
    """
    from morphis.elements.vector import Vector

    if len(indexed_tensors) < 2:
        raise ValueError("Contraction requires at least 2 indexed tensors")

    # Collect all index information
    all_indices = [it.indices for it in indexed_tensors]
    all_data = [it.tensor.data for it in indexed_tensors]

    # Count index occurrences to determine output indices
    index_counts: dict[str, int] = {}
    for indices in all_indices:
        for idx in indices:
            index_counts[idx] = index_counts.get(idx, 0) + 1

    # Output indices are those that appear exactly once (not contracted), in
    # order of first appearance, lot before geometric to match Vector storage
    free = [idx for indices in all_indices for idx in indices if index_counts[idx] == 1]
    geo_indices = {idx for it in indexed_tensors for idx in it.output_geo_indices}
    output_indices = "".join(sorted(free, key=lambda idx: idx in geo_indices))

    # Build einsum signature
    input_sig = ",".join(all_indices)
    einsum_sig = f"{input_sig}->{output_indices}"

    # Perform contraction
    result_data = einsum(einsum_sig, *all_data)

    # Get metric from first tensor that has one
    metric = None
    for it in indexed_tensors:
        if hasattr(it.tensor, "metric") and it.tensor.metric is not None:
            metric = it.tensor.metric
            break

    # Infer grade from output
    result_grade = _infer_grade_from_indexed(indexed_tensors, output_indices)

    return Vector(data=result_data, grade=result_grade, metric=metric)


def _output_geo_indices(tensor: "Vector | Operator", indices: str) -> str:
    """
    The labels, among one per axis of the tensor, of its output geometric axes.

    A Vector is stored (*lot, *geo) and all its geometric axes are output. An
    Operator is stored (*out_lot, *in_lot, *out_geo, *in_geo).
    """
    from morphis.elements.vector import Vector
    from morphis.operations.operator import Operator

    result = ""

    if isinstance(tensor, Vector):
        result = indices[len(tensor.lot) :]

    elif isinstance(tensor, Operator):
        start = tensor.output_spec.collection + tensor.input_spec.collection
        result = indices[start : start + tensor.output_spec.grade]

    return result


def _infer_grade_from_indexed(indexed_tensors: tuple[IndexedTensor, ...], output_indices: str) -> int:
    """
    Infer grade for IndexedTensor contraction result.

    The result grade equals the number of output geometric indices that appear
    in the result. Each IndexedTensor carries its output_geo_indices, which
    identify which indices represent geometric (grade-contributing) dimensions.
    """
    # Collect output geometric indices from all tensors
    all_output_geo = set()
    for it in indexed_tensors:
        all_output_geo.update(it.output_geo_indices)

    # Result grade = count of output geometric indices in the result
    return sum(1 for idx in output_indices if idx in all_output_geo)


# =============================================================================
# contract() - Einsum-Style API
# =============================================================================


def contract(signature: str, *tensors: "Vector | Operator") -> "Vector":
    """
    Einsum-style contraction for Morphis tensors.

    Works exactly like numpy.einsum, but accepts Vector and Operator objects.
    Extracts the underlying data, performs the einsum, and wraps the result
    back into a Vector.

    Args:
        signature: Einsum signature string (e.g., "mn, n -> m")
        *tensors: Morphis objects (Vector or Operator) to contract

    Returns:
        Vector containing the contracted result

    Examples:
        >>> g = euclidean_metric(3)
        >>> u = Vector([1, 2, 3], grade=1, metric=g)
        >>> v = Vector([4, 5, 6], grade=1, metric=g)

        >>> # Dot product
        >>> s = contract("a, a ->", u, v)
        >>> s.data  # 1*4 + 2*5 + 3*6 = 32

        >>> # Matrix-vector product
        >>> M = Vector(data, grade=2, metric=g)  # shape (3, 3)
        >>> w = contract("ab, b -> a", M, v)

        >>> # Outer product
        >>> outer = contract("a, b -> ab", u, v)

        >>> # Batch contraction
        >>> G = Vector(data, grade=2, lot=(M, N), metric=g)  # shape (M, N, 3, 3)
        >>> q = Vector(data, grade=0, lot=(N,), metric=g)  # shape (N,)
        >>> b = contract("mnab, n -> mab", G, q)
    """
    from morphis.elements.vector import Vector

    if len(tensors) < 1:
        raise ValueError("contract() requires at least 1 tensor")

    # Extract data arrays from tensors
    data_arrays = [t.data for t in tensors]

    # Normalize signature: allow spaces around comma and arrow
    sig = signature.replace(" ", "")

    # Perform einsum
    result_data = einsum(sig, *data_arrays)

    # Get metric from first tensor that has one
    metric = None
    for t in tensors:
        if hasattr(t, "metric") and t.metric is not None:
            metric = t.metric
            break

    # Infer grade from output shape
    result_grade = _infer_grade_from_signature(sig, tensors)

    return Vector(data=result_data, grade=result_grade, metric=metric)


def _infer_grade_from_signature(signature: str, tensors: tuple) -> int:
    """
    Infer grade for einsum-style contraction result.

    The result grade equals the number of output geometric indices that appear
    in the result, as it does for the bracket syntax.
    """
    if "->" in signature:
        input_part, output_indices = signature.split("->")
    else:
        # einsum's implicit output: indices that appear exactly once, sorted
        input_part = signature
        letters = input_part.replace(",", "")
        output_indices = "".join(sorted(idx for idx in set(letters) if letters.count(idx) == 1))

    geo_indices = set()
    for indices, t in zip(input_part.split(","), tensors, strict=False):
        geo_indices.update(_output_geo_indices(t, indices))

    # Count geometric indices in output
    result_grade = sum(1 for idx in output_indices if idx in geo_indices)

    return result_grade
