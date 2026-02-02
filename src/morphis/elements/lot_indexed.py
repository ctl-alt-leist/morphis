"""
Geometric Algebra - Lot Indexed Vectors

LotIndexed provides explicit broadcasting semantics for lot dimensions
using index labels. This enables einsum-style operations over collection
dimensions while preserving the geometric structure.

Semantics:
- Shared indices: element-wise for +, -, /, ^ | contraction for *
- Non-shared indices: outer product

Examples:
    x = Vector(...)  # lot (M,)
    y = Vector(...)  # lot (N, K)

    # Outer product on lot dimensions
    r = y["nk"] - x["m"]                 # lot (N, K, M)

    # Reorder to desired lot order
    r = (y["nk"] - x["m"])["mnk"]        # lot (M, N, K)

    # Contraction with *
    b = G["mn"] * q["n"]                 # n contracts -> lot (M,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy import einsum, ndarray


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


@dataclass(slots=True)
class LotIndexed:
    """
    Lightweight wrapper pairing a Vector with lot index labels.

    Enables explicit broadcasting over lot dimensions without affecting
    geometric structure. The indices label only lot dimensions, not
    geometric dimensions.

    Attributes:
        vector: The underlying Vector
        indices: String of index labels, one per lot dimension

    Examples:
        >>> v = Vector(data, grade=1, metric=g)  # lot=(M, N)
        >>> vi = v["mn"]  # LotIndexed with lot indices "mn"
        >>> vi.vector is v
        True
    """

    vector: "Vector"
    indices: str

    def __post_init__(self):
        """Validate index count matches lot dimensions."""
        n_lot = len(self.vector.lot)
        if len(self.indices) != n_lot:
            raise ValueError(
                f"Index string '{self.indices}' has {len(self.indices)} indices, but vector has {n_lot} lot dimensions"
            )

        # Check for duplicate indices
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"Duplicate indices in '{self.indices}'")

    def __getitem__(self, new_indices: str) -> "LotIndexed":
        """
        Reorder lot dimensions to match new index order.

        Args:
            new_indices: New ordering of index labels

        Returns:
            LotIndexed with reordered lot dimensions

        Examples:
            >>> r = (y["nk"] - x["m"])["mnk"]  # reorder to (M, N, K)
        """
        if set(new_indices) != set(self.indices):
            raise ValueError(f"Index mismatch: '{new_indices}' must be a permutation of '{self.indices}'")

        if new_indices == self.indices:
            return self

        # Build axis permutation for lot dimensions only
        # Axes: lot_axes + geo_axes
        n_lot = len(self.indices)
        n_geo = self.vector.grade

        # Map from current index -> position
        current_pos = {idx: i for i, idx in enumerate(self.indices)}
        # New positions for lot axes
        lot_perm = [current_pos[idx] for idx in new_indices]
        # Geo axes stay in place (after lot axes)
        geo_axes = list(range(n_lot, n_lot + n_geo))
        full_perm = lot_perm + geo_axes

        # Transpose data
        new_data = self.vector.data.transpose(full_perm)

        from morphis.elements.vector import Vector

        new_vector = Vector(
            data=new_data,
            grade=self.vector.grade,
            metric=self.vector.metric,
        )
        return LotIndexed(new_vector, new_indices)

    # =========================================================================
    # Arithmetic Operations (element-wise on shared, outer on non-shared)
    # =========================================================================

    def __add__(self, other: "LotIndexed") -> "LotIndexed":
        """Add with lot broadcasting."""
        return _lot_broadcast_binary(self, other, "add")

    def __radd__(self, other: "LotIndexed") -> "LotIndexed":
        if not isinstance(other, LotIndexed):
            return NotImplemented
        return _lot_broadcast_binary(other, self, "add")

    def __sub__(self, other: "LotIndexed") -> "LotIndexed":
        """Subtract with lot broadcasting."""
        return _lot_broadcast_binary(self, other, "sub")

    def __rsub__(self, other: "LotIndexed") -> "LotIndexed":
        if not isinstance(other, LotIndexed):
            return NotImplemented
        return _lot_broadcast_binary(other, self, "sub")

    def __truediv__(self, other: "LotIndexed | ndarray | float") -> "LotIndexed":
        """Divide with lot broadcasting."""
        if isinstance(other, LotIndexed):
            return _lot_broadcast_binary(self, other, "div")
        # Scalar division
        from morphis.elements.vector import Vector

        new_vector = Vector(
            data=self.vector.data / other,
            grade=self.vector.grade,
            metric=self.vector.metric,
        )
        return LotIndexed(new_vector, self.indices)

    def __mul__(self, other: "LotIndexed | ndarray | float") -> "LotIndexed":
        """
        Multiplication: contraction on shared indices (Einstein convention).

        For scalar/array multiplication, use standard broadcasting.
        """
        if isinstance(other, LotIndexed):
            return _lot_contract(self, other)
        # Scalar multiplication
        from morphis.elements.vector import Vector

        new_vector = Vector(
            data=self.vector.data * other,
            grade=self.vector.grade,
            metric=self.vector.metric,
        )
        return LotIndexed(new_vector, self.indices)

    def __rmul__(self, other: "LotIndexed | ndarray | float") -> "LotIndexed":
        if isinstance(other, LotIndexed):
            return _lot_contract(other, self)
        # Check for numeric types (scalar/array)
        if isinstance(other, (int, float, complex, ndarray)):
            from morphis.elements.vector import Vector

            new_vector = Vector(
                data=other * self.vector.data,
                grade=self.vector.grade,
                metric=self.vector.metric,
            )
            return LotIndexed(new_vector, self.indices)
        # Unknown type - let Python try other options
        return NotImplemented

    def __and__(self, other: "LotIndexed") -> "LotIndexed":
        """Hadamard (element-wise) multiplication on shared lot indices."""
        return _lot_broadcast_binary(self, other, "mul")

    def __xor__(self, other: "LotIndexed") -> "LotIndexed":
        """Wedge product with lot broadcasting."""
        return _lot_broadcast_binary(self, other, "wedge")

    def __pow__(self, exponent: int) -> "LotIndexed":
        """Power operation preserving indices."""
        from morphis.elements.vector import Vector

        new_vector = Vector(
            data=self.vector.data**exponent,
            grade=self.vector.grade,
            metric=self.vector.metric,
        )
        return LotIndexed(new_vector, self.indices)

    # =========================================================================
    # Utility
    # =========================================================================

    def norm(self) -> "LotIndexed":
        """Compute norm, preserving lot indices."""
        # norm() returns NDArray with shape matching lot
        norm_data = self.vector.norm()

        from morphis.elements.vector import Vector

        # Wrap as grade-0 vector to preserve lot structure
        new_vector = Vector(
            data=norm_data,
            grade=0,
            metric=self.vector.metric,
        )
        return LotIndexed(new_vector, self.indices)

    def sum(self, axis: int | None = None) -> "LotIndexed":
        """Sum over lot axis, removing that index."""
        if axis is None:
            # Sum over all -> scalar, no lot indices
            from morphis.elements.vector import Vector

            new_vector = Vector(
                data=self.vector.data.sum(axis=tuple(range(len(self.indices)))),
                grade=self.vector.grade,
                metric=self.vector.metric,
            )
            return LotIndexed(new_vector, "")

        # Remove the index at the summed axis
        new_indices = self.indices[:axis] + self.indices[axis + 1 :]
        summed_vector = self.vector.sum(axis=axis)

        return LotIndexed(summed_vector, new_indices)

    def __repr__(self) -> str:
        return f"LotIndexed({self.vector!r}, indices='{self.indices}')"


# =============================================================================
# Internal Operations
# =============================================================================


def _compute_broadcast_info(left_indices: str, right_indices: str) -> tuple[str, list[int], list[int]]:
    """
    Compute broadcast information for two indexed operands.

    Returns:
        result_indices: Output index string (non-shared left, non-shared right, shared)
        left_expand: Axes to add to left operand (via np.newaxis)
        right_expand: Axes to add to right operand (via np.newaxis)
    """
    left_set = set(left_indices)
    right_set = set(right_indices)

    shared = left_set & right_set
    left_only = [i for i in left_indices if i not in shared]
    right_only = [i for i in right_indices if i not in shared]
    shared_list = [i for i in left_indices if i in shared]

    # Result order: left_only, right_only, shared (as they appear in left)
    result_indices = "".join(left_only + right_only + shared_list)

    return result_indices, left_only, right_only, shared_list


def _lot_broadcast_binary(left: LotIndexed, right: LotIndexed, op: str) -> LotIndexed:
    """
    Perform binary operation with lot broadcasting.

    Shared indices: element-wise
    Non-shared indices: outer product
    """

    from morphis.elements.metric import Metric
    from morphis.elements.vector import Vector
    from morphis.operations.products import wedge

    if not isinstance(right, LotIndexed):
        raise TypeError(f"Expected LotIndexed, got {type(right)}")

    left_indices = left.indices
    right_indices = right.indices

    # Handle empty indices (scalars)
    if not left_indices and not right_indices:
        # Both are scalars, just operate
        if op == "add":
            result_vector = left.vector + right.vector
        elif op == "sub":
            result_vector = left.vector - right.vector
        elif op == "mul":
            result_data = left.vector.data * right.vector.data
            metric = Metric.merge(left.vector.metric, right.vector.metric)
            result_vector = Vector(data=result_data, grade=left.vector.grade, metric=metric)
        elif op == "div":
            result_vector = left.vector / right.vector.data
        elif op == "wedge":
            result_vector = wedge(left.vector, right.vector)
        else:
            raise ValueError(f"Unknown operation: {op}")
        return LotIndexed(result_vector, "")

    left_set = set(left_indices)
    right_set = set(right_indices)
    shared = left_set & right_set

    # Build result index order: left_only + right_only + shared
    left_only = [i for i in left_indices if i not in shared]
    right_only = [i for i in right_indices if i not in shared]
    shared_ordered = [i for i in left_indices if i in shared]
    result_indices = "".join(left_only + right_only + shared_ordered)

    # Determine geo dimensions
    n_left_geo = left.vector.grade
    n_right_geo = right.vector.grade

    # Create unique letters for geo dimensions (use uppercase to avoid conflicts)
    left_geo_labels = "".join(chr(ord("A") + i) for i in range(n_left_geo))
    right_geo_labels = "".join(chr(ord("A") + n_left_geo + i) for i in range(n_right_geo))

    # For wedge, geo dimensions combine; for others, they must match (unless one is scalar)
    if op == "wedge":
        result_geo_labels = left_geo_labels + right_geo_labels
    elif op == "div" and n_right_geo == 0:
        # Division by scalar: broadcast scalar over all geo dimensions
        result_geo_labels = left_geo_labels
        right_geo_labels = ""  # Scalar has no geo labels
    elif op in ("mul", "div") and n_left_geo == 0:
        # Scalar times/by something: result takes right's geo
        result_geo_labels = right_geo_labels
        left_geo_labels = ""
    else:
        # For add/sub, geo dimensions must match
        if n_left_geo != n_right_geo:
            raise ValueError(f"Grade mismatch for {op}: {left.vector.grade} vs {right.vector.grade}")
        result_geo_labels = left_geo_labels
        right_geo_labels = left_geo_labels  # They share the same geo labels

    # Build einsum signature
    left_sig = left_indices + left_geo_labels
    right_sig = right_indices + right_geo_labels
    result_sig = result_indices + result_geo_labels

    left_data = left.vector.data
    right_data = right.vector.data

    # Merge metrics
    metric = Metric.merge(left.vector.metric, right.vector.metric)

    # Perform the operation
    if op == "add":
        # Use einsum for broadcasting, then add
        result_data = einsum(f"{left_sig},{right_sig}->{result_sig}", left_data, right_data * 0) + einsum(
            f"{left_sig},{right_sig}->{result_sig}", left_data * 0 + 1, right_data
        )
        # Simpler: broadcast manually
        result_data = _broadcast_and_operate(
            left_data,
            right_data,
            left_indices,
            right_indices,
            result_indices,
            n_left_geo,
            n_right_geo,
            lambda a, b: a + b,
        )
        result_grade = left.vector.grade
    elif op == "sub":
        result_data = _broadcast_and_operate(
            left_data,
            right_data,
            left_indices,
            right_indices,
            result_indices,
            n_left_geo,
            n_right_geo,
            lambda a, b: a - b,
        )
        result_grade = left.vector.grade
    elif op == "mul":
        # Hadamard (element-wise)
        result_data = _broadcast_and_operate(
            left_data,
            right_data,
            left_indices,
            right_indices,
            result_indices,
            n_left_geo,
            n_right_geo,
            lambda a, b: a * b,
        )
        result_grade = left.vector.grade
    elif op == "div":
        result_data = _broadcast_and_operate(
            left_data,
            right_data,
            left_indices,
            right_indices,
            result_indices,
            n_left_geo,
            n_right_geo,
            lambda a, b: a / b,
        )
        result_grade = left.vector.grade
    elif op == "wedge":
        # For wedge, we need the actual wedge product, not just data manipulation
        # Expand lot dimensions first, then compute wedge
        left_expanded, right_expanded = _expand_for_broadcast(
            left_data, right_data, left_indices, right_indices, result_indices, n_left_geo, n_right_geo
        )
        # Create expanded vectors and compute wedge
        left_vec = Vector(data=left_expanded, grade=left.vector.grade, metric=metric)
        right_vec = Vector(data=right_expanded, grade=right.vector.grade, metric=metric)
        result_vector = wedge(left_vec, right_vec)
        return LotIndexed(result_vector, result_indices)
    else:
        raise ValueError(f"Unknown operation: {op}")

    result_vector = Vector(data=result_data, grade=result_grade, metric=metric)
    return LotIndexed(result_vector, result_indices)


def _broadcast_and_operate(
    left_data,
    right_data,
    left_indices: str,
    right_indices: str,
    result_indices: str,
    n_left_geo: int,
    n_right_geo: int,
    op_func,
):
    """
    Broadcast two arrays over lot dimensions and apply operation.
    """
    from numpy import newaxis

    # Build the expanded arrays
    left_expanded, right_expanded = _expand_for_broadcast(
        left_data, right_data, left_indices, right_indices, result_indices, n_left_geo, n_right_geo
    )

    # If geo dimensions differ, we need to add trailing dimensions for broadcasting
    # This handles scalar division/multiplication: (M, N, K, 3, 3) / (M, N, K) -> need (M, N, K, 1, 1)
    if n_left_geo > n_right_geo:
        for _ in range(n_left_geo - n_right_geo):
            right_expanded = right_expanded[..., newaxis]
    elif n_right_geo > n_left_geo:
        for _ in range(n_right_geo - n_left_geo):
            left_expanded = left_expanded[..., newaxis]

    return op_func(left_expanded, right_expanded)


def _expand_for_broadcast(
    left_data, right_data, left_indices: str, right_indices: str, result_indices: str, n_left_geo: int, n_right_geo: int
):
    """
    Expand arrays to have compatible shapes for broadcasting.

    Result axes order: result_lot_indices + geo_axes
    """
    from numpy import expand_dims

    # Expand left: add newaxis for indices in result but not in left
    left_shape_map = {idx: i for i, idx in enumerate(left_indices)}
    left_perm = []
    left_expand_axes = []

    for i, idx in enumerate(result_indices):
        if idx in left_shape_map:
            left_perm.append(left_shape_map[idx])
        else:
            left_expand_axes.append(i)

    # Similarly for right
    right_shape_map = {idx: i for i, idx in enumerate(right_indices)}
    right_perm = []
    right_expand_axes = []

    for i, idx in enumerate(result_indices):
        if idx in right_shape_map:
            right_perm.append(right_shape_map[idx])
        else:
            right_expand_axes.append(i)

    # Reorder left lot dimensions to match result order (for indices that exist)
    # Then add geo dimensions
    left_lot_perm = left_perm + list(range(len(left_indices), len(left_indices) + n_left_geo))
    left_reordered = (
        left_data.transpose(left_lot_perm) if left_lot_perm != list(range(len(left_lot_perm))) else left_data
    )

    # Insert newaxis for missing dimensions
    for ax in sorted(left_expand_axes):
        left_reordered = expand_dims(left_reordered, axis=ax)

    # Same for right
    right_lot_perm = right_perm + list(range(len(right_indices), len(right_indices) + n_right_geo))
    right_reordered = (
        right_data.transpose(right_lot_perm) if right_lot_perm != list(range(len(right_lot_perm))) else right_data
    )

    for ax in sorted(right_expand_axes):
        right_reordered = expand_dims(right_reordered, axis=ax)

    return left_reordered, right_reordered


def _lot_contract(left: LotIndexed, right: LotIndexed) -> LotIndexed:
    """
    Contract two LotIndexed tensors on shared lot indices (Einstein convention).

    Shared indices are summed over (contracted).
    Non-shared indices form the outer product.
    """
    from morphis.elements.metric import Metric
    from morphis.elements.vector import Vector

    left_indices = left.indices
    right_indices = right.indices

    left_set = set(left_indices)
    right_set = set(right_indices)
    shared = left_set & right_set

    # Result indices: non-shared only (shared are contracted away)
    left_only = [i for i in left_indices if i not in shared]
    right_only = [i for i in right_indices if i not in shared]
    result_indices = "".join(left_only + right_only)

    n_left_geo = left.vector.grade
    n_right_geo = right.vector.grade

    # For contraction, we treat this as lot-level contraction
    # Geo dimensions must match and stay the same
    if n_left_geo != n_right_geo:
        raise ValueError(f"Grade mismatch for contraction: {left.vector.grade} vs {right.vector.grade}")

    # Build einsum signature
    # Lot indices are labeled with the index string
    # Geo indices use uppercase letters
    geo_labels = "".join(chr(ord("A") + i) for i in range(n_left_geo))

    left_sig = left_indices + geo_labels
    right_sig = right_indices + geo_labels
    result_sig = result_indices + geo_labels

    einsum_sig = f"{left_sig},{right_sig}->{result_sig}"

    result_data = einsum(einsum_sig, left.vector.data, right.vector.data)

    metric = Metric.merge(left.vector.metric, right.vector.metric)
    result_vector = Vector(data=result_data, grade=left.vector.grade, metric=metric)

    return LotIndexed(result_vector, result_indices)
