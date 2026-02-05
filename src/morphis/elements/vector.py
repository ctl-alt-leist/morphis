"""
Geometric Algebra - Vector

A Vector represents a homogeneous multivector of pure grade k in geometric algebra.
Storage shape is (*lot, *geo) where geo is (dim,) * grade.

Vectors are antisymmetric (k,0)-tensors. A Vector that can be factorized as
v1 ^ v2 ^ ... ^ vk is called a "blade" (simple k-vector) and represents a
k-dimensional oriented subspace.

Every Vector requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from numpy import asarray, broadcast_shapes, stack, zeros
from numpy.typing import NDArray
from pydantic import ConfigDict, model_validator

from morphis.elements.base import IndexableMixin
from morphis.elements.metric import Metric
from morphis.elements.tensor import Tensor


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Accessor Classes for Slicing
# =============================================================================


class AtAccessor:
    """
    Accessor for lot-only (collection) slicing on a Vector.

    Usage:
        v.at[0]        # Slice first lot dimension
        v.at[:, 1:]    # Slice first two lot dimensions

    Indexing through .at only affects lot dimensions; geometric dimensions
    are preserved with implicit slice(None).
    """

    __slots__ = ("_vector",)

    def __init__(self, vector: "Vector"):
        self._vector = vector

    def __getitem__(self, index) -> "Vector":
        """Slice lot dimensions only, preserving geometric structure."""
        # Normalize index to tuple
        if not isinstance(index, tuple):
            index = (index,)

        n_lot = len(self._vector.lot)
        n_geo = self._vector.grade

        # Validate: can't specify more indices than lot dims
        if len(index) > n_lot:
            raise IndexError(
                f"Too many indices for lot dimensions: got {len(index)}, but vector has {n_lot} lot dimensions"
            )

        # Pad with slice(None) to cover remaining lot dims
        if len(index) < n_lot:
            index = index + (slice(None),) * (n_lot - len(index))

        # Add slice(None) for all geometric axes
        full_index = index + (slice(None),) * n_geo

        new_data = self._vector.data[full_index]
        return Vector(new_data, grade=self._vector.grade, metric=self._vector.metric)

    def __repr__(self) -> str:
        return f"AtAccessor(lot={self._vector.lot})"


class OnAccessor:
    """
    Accessor for geo-only (geometric) slicing on a Vector.

    Usage:
        v.on[0]        # Slice first geometric dimension
        v.on[0, 1]     # Extract component v^{01} for a bivector

    Indexing through .on only affects geometric dimensions; lot dimensions
    are preserved with implicit slice(None).
    """

    __slots__ = ("_vector",)

    def __init__(self, vector: "Vector"):
        self._vector = vector

    def __getitem__(self, index) -> "Vector":
        """Slice geometric dimensions only, preserving lot structure."""
        # Normalize index to tuple
        if not isinstance(index, tuple):
            index = (index,)

        n_lot = len(self._vector.lot)
        n_geo = self._vector.grade

        # Validate: can't specify more indices than geo dims
        if len(index) > n_geo:
            raise IndexError(
                f"Too many indices for geometric dimensions: got {len(index)}, "
                f"but vector has {n_geo} geometric dimensions (grade={self._vector.grade})"
            )

        # Build full index: slice(None) for lot dims, then the geo indices
        full_index = (slice(None),) * n_lot + index

        new_data = self._vector.data[full_index]

        # Compute new grade: each integer index reduces grade by 1
        new_grade = n_geo
        for idx in index:
            if isinstance(idx, int):
                new_grade -= 1

        return Vector(new_data, grade=new_grade, metric=self._vector.metric)

    def __repr__(self) -> str:
        return f"OnAccessor(geo={self._vector.geo})"


# =============================================================================
# Vector Class
# =============================================================================


class Vector(IndexableMixin, Tensor):
    """
    A Vector (k-vector) in geometric algebra.

    Represents a homogeneous multivector of pure grade k. Storage shape is
    (*lot, *geo) where geo is (dim,) * grade.
    Scalars have grade=0, grade-1 vectors, bivectors grade=2, etc.

    The components V^{m_1 ... m_k} are stored with full redundancy (all d^k
    elements), satisfying antisymmetry: V^{...m...n...} = -V^{...n...m...}.

    A Vector that can be factorized as v1 ^ v2 ^ ... ^ vk is called a "blade"
    (simple k-vector). Use the .is_blade property to check. Not every k-vector
    is a blade - for example, e12 + e34 in 4D cannot be factorized.

    Attributes:
        data: The underlying array of components (inherited)
        grade: The grade (0=scalar, 1=vector, 2=bivector, etc.)
        metric: The complete geometric context (inherited)
        lot: Shape of the lot (collection) dimensions (inherited)

    Examples:
        >>> from morphis.elements.metric import euclidean_metric
        >>> m = euclidean_metric(3)
        >>> v = Vector([1, 0, 0], grade=1, metric=m)
        >>> v.dim
        3
        >>> v.is_blade
        True
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # Vector-specific attribute
    grade: int

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(self, data=None, /, **kwargs):
        """Allow positional argument for data: Vector(arr, grade=1, metric=m)."""
        if data is not None:
            kwargs["data"] = data

        # Set contravariant from grade, covariant always 0 for Vectors
        if "grade" in kwargs:
            kwargs["contravariant"] = kwargs["grade"]
        kwargs["covariant"] = 0

        super().__init__(**kwargs)

    @classmethod
    def stack(cls, vectors: Sequence["Vector"], axis: int = 0) -> "Vector":
        """
        Stack Vectors along a new lot axis.

        Args:
            vectors: Sequence of Vectors with same grade and metric
            axis: Lot axis for stacking (default 0, prepends new dimension)

        Returns:
            Vector with new lot dimension at specified axis

        Examples:
            >>> from morphis.elements import Vector, euclidean_metric
            >>> from numpy.random import randn
            >>> g = euclidean_metric(3)
            >>> vectors = [Vector(randn(3), grade=1, metric=g) for _ in range(10)]
            >>> stacked = Vector.stack(vectors, axis=0)
            >>> stacked.lot
            (10,)
        """
        if not vectors:
            raise ValueError("Cannot stack empty sequence")

        first = vectors[0]
        grade = first.grade
        metric = first.metric

        for v in vectors[1:]:
            if v.grade != grade:
                raise ValueError(f"Grade mismatch: {v.grade} != {grade}")
            metric = Metric.merge(metric, v.metric)

        data = stack([v.data for v in vectors], axis=axis)
        return cls(data, grade=grade, metric=metric)

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def _sync_grade_contravariant(self):
        """Ensure grade and contravariant stay in sync."""
        # grade should equal contravariant for Vectors
        if hasattr(self, "contravariant") and self.grade != self.contravariant:
            object.__setattr__(self, "grade", self.contravariant)
        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_blade(self) -> bool:
        """
        Check if this Vector is a blade (simple k-vector).

        A blade can be factorized as v1 ^ v2 ^ ... ^ vk, representing a
        k-dimensional oriented subspace. Not all k-vectors are blades.

        For grades 0 and 1, all Vectors are blades.
        For grade 2+, checks if the Vector is factorizable.

        Returns:
            True if this Vector is a simple k-vector (vec)
        """
        # Scalars and grade-1 vectors are always blades
        if self.grade <= 1:
            return True

        # For higher grades, use factorization to check
        # A k-vector is a blade iff it has rank 1 as an antisymmetric tensor
        from morphis.operations.factorization import factor

        try:
            factor(self)
            return True
        except ValueError:
            return False

    @property
    def at(self) -> AtAccessor:
        """
        Accessor for lot-only (collection) slicing.

        Usage:
            v.at[0]        # Slice first lot dimension
            v.at[:, 1:]    # Slice first two lot dimensions

        Indexing through .at only affects lot dimensions; geometric dimensions
        are preserved with implicit slice(None).
        """
        return AtAccessor(self)

    @property
    def on(self) -> OnAccessor:
        """
        Accessor for geo-only (geometric) slicing.

        Usage:
            v.on[0]        # Slice first geometric dimension
            v.on[0, 1]     # Extract component v^{01} for a bivector

        Indexing through .on only affects geometric dimensions; lot dimensions
        are preserved with implicit slice(None).
        """
        return OnAccessor(self)

    @property
    def real(self) -> "Vector":
        """
        Return Vector with real part of coefficients.

        For real Vectors, returns a copy.
        For complex Vectors (phasors), extracts real part.

        Returns:
            Vector with real-valued coefficients

        Examples:
            >>> v = Vector([1 + 2j, 3 - 4j, 5j], grade=1, metric=g)
            >>> v.real.data
            array([1., 3., 0.])
        """
        return Vector(
            data=self.data.real.copy(),
            grade=self.grade,
            metric=self.metric,
        )

    @property
    def imag(self) -> "Vector":
        """
        Return Vector with imaginary part of coefficients.

        For real Vectors, returns zero Vector.
        For complex Vectors (phasors), extracts imaginary part.

        Returns:
            Vector with imaginary coefficients as real values

        Examples:
            >>> v = Vector([1 + 2j, 3 - 4j, 5j], grade=1, metric=g)
            >>> v.imag.data
            array([2., -4., 5.])
        """
        return Vector(
            data=self.data.imag.copy(),
            grade=self.grade,
            metric=self.metric,
        )

    # =========================================================================
    # Indexing (implements IndexableMixin interface)
    # =========================================================================

    def _index(self, indices: str):
        """
        Create an indexed wrapper for einsum-style operations.

        If indices match lot dimensions only, returns LotIndexed for
        lot-level broadcasting. If indices match all dimensions (lot + geo),
        returns IndexedTensor for full tensor contraction.

        Args:
            indices: String of index labels

        Returns:
            LotIndexed if len(indices) == len(lot)
            IndexedTensor if len(indices) == len(lot) + grade

        Examples:
            v = Vector(data, grade=2)  # lot=(M, N), shape=(M, N, 3, 3)
            v["mn"]    # LotIndexed for lot broadcasting
            v["mnab"]  # IndexedTensor for full tensor contraction
        """
        n_lot = len(self.lot)
        n_total = self.data.ndim

        if len(indices) == n_lot:
            # Lot-only indexing -> LotIndexed
            from morphis.elements.lot_indexed import LotIndexed

            return LotIndexed(self, indices)
        elif len(indices) == n_total:
            # Full indexing -> IndexedTensor
            # All geometric indices (after lot) are "output geometric" for Vectors
            from morphis.algebra.contraction import IndexedTensor

            output_geo_indices = indices[n_lot:]
            return IndexedTensor(self, indices, output_geo_indices=output_geo_indices)
        else:
            raise ValueError(
                f"Index string '{indices}' has {len(indices)} indices, but expected "
                f"{n_lot} (lot only) or {n_total} (lot + geo)"
            )

    def _slice(self, index) -> "Vector":
        """
        Slice into the Vector, returning a new Vector.

        Indexing follows numpy semantics but always returns a Vector.
        The index is applied positionally to the full shape (lot + geo).
        Lot and geo axes are tracked independently:
        - Integer indices reduce the corresponding lot or geo axis
        - Slice indices preserve the axis with potentially different size

        Examples:
            v = Vector(data, grade=2, metric=m)  # lot=(M, N), geo=(3, 3)

            v[0]           # lot=(N,), geo=(3, 3) - index first lot axis
            v[:, :, 0]     # lot=(M, N), geo=(3,) - index first geo axis

        For explicit lot-only or geo-only indexing, use v.at[...] or v.on[...].
        """
        # Normalize index to tuple
        if not isinstance(index, tuple):
            index = (index,)

        n_lot = len(self.lot)
        n_geo = self.grade

        # Apply index to data
        new_data = self.data[index]

        # Determine new grade by counting integer indices in geo positions
        # First, figure out which indices apply to lot vs geo
        # Handle ellipsis expansion
        expanded_index = self._expand_index(index, n_lot + n_geo)

        new_grade = n_geo
        for i, idx in enumerate(expanded_index):
            if i >= n_lot:  # This is a geo axis
                if isinstance(idx, int):
                    new_grade -= 1

        return Vector(new_data, grade=new_grade, metric=self.metric)

    def _expand_index(self, index: tuple, ndim: int) -> tuple:
        """Expand ellipsis in index to explicit slices."""
        # Count non-ellipsis elements
        n_non_ellipsis = sum(1 for idx in index if idx is not ...)
        n_ellipsis = sum(1 for idx in index if idx is ...)

        if n_ellipsis > 1:
            raise IndexError("Only one ellipsis allowed in index")

        if n_ellipsis == 0:
            # No ellipsis - pad with slice(None) to match ndim
            if len(index) < ndim:
                return index + (slice(None),) * (ndim - len(index))
            return index

        # Expand ellipsis
        n_to_expand = ndim - n_non_ellipsis
        expanded = []
        for idx in index:
            if idx is ...:
                expanded.extend([slice(None)] * n_to_expand)
            else:
                expanded.append(idx)
        return tuple(expanded)

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other: Vector) -> Vector:
        """Add two Vectors of the same grade."""
        if not isinstance(other, Vector):
            raise TypeError(f"Cannot add Vector and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot add Vectors of grade {self.grade} and {other.grade}")

        metric = Metric.merge(self.metric, other.metric)
        lot = broadcast_shapes(self.lot, other.lot)

        return Vector(
            data=self.data + other.data,
            grade=self.grade,
            metric=metric,
            lot=lot,
        )

    def __sub__(self, other: Vector) -> Vector:
        """Subtract two Vectors of the same grade."""
        if not isinstance(other, Vector):
            raise TypeError(f"Cannot subtract Vector and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot subtract Vectors of grade {self.grade} and {other.grade}")

        metric = Metric.merge(self.metric, other.metric)
        lot = broadcast_shapes(self.lot, other.lot)

        return Vector(
            data=self.data - other.data,
            grade=self.grade,
            metric=metric,
            lot=lot,
        )

    def __mul__(self, other) -> Vector | MultiVector:
        """Multiplication: scalar or geometric product.

        - Scalar: returns Vector with scaled data
        - Vector/MultiVector: returns geometric product (MultiVector)
        - Grade-0 Vector * Frame/Operator: delegates to other's __rmul__
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector
        from morphis.operations.operator import Operator

        if isinstance(other, Vector):
            from morphis.operations.products import geometric

            return geometric(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _geometric_v_mv

            return _geometric_v_mv(self, other)
        elif isinstance(other, Operator):
            # Grade-0 Vector (scalar) can multiply Operator via Operator.__rmul__
            if self.grade == 0 and self.lot == ():
                return NotImplemented
            raise TypeError("Vector * Operator not currently supported (use L * v)")
        elif isinstance(other, Frame):
            # Delegate to Frame.__rmul__ which handles:
            # - grade-0 scalar: returns scaled Frame
            # - other grades: returns geometric product (MultiVector)
            return NotImplemented
        else:
            # Scalar or array multiplication
            other = asarray(other)

            # Broadcast over geometric axes if needed
            if other.ndim > 0 and other.ndim < self.data.ndim:
                for _ in range(self.grade):
                    other = other[..., None]

            return Vector(
                data=self.data * other,
                grade=self.grade,
                metric=self.metric,
            )

    def __rmul__(self, other) -> Vector | MultiVector:
        """Right multiplication: scalar or geometric product."""
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Vector):
            from morphis.operations.products import geometric

            return geometric(other, self)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _geometric_mv_v

            return _geometric_mv_v(other, self)
        else:
            # Scalar or array multiplication (commutative)
            other = asarray(other)

            # Broadcast over geometric axes if needed
            if other.ndim > 0 and other.ndim < self.data.ndim:
                for _ in range(self.grade):
                    other = other[..., None]

            return Vector(
                data=self.data * other,
                grade=self.grade,
                metric=self.metric,
            )

    def __truediv__(self, other) -> Vector:
        """
        Scalar or array division.

        If other is an array with shape matching lot dimensions, it broadcasts
        over the geometric axes automatically.

        Examples:
            v / 2.0           # scalar division
            v / magnitudes    # magnitudes.shape == v.lot, broadcasts over geo
        """
        other = asarray(other)

        # Broadcast over geometric axes if needed
        if other.ndim > 0 and other.ndim < self.data.ndim:
            # Add trailing dimensions for geo axes
            for _ in range(self.grade):
                other = other[..., None]

        return Vector(
            data=self.data / other,
            grade=self.grade,
            metric=self.metric,
        )

    def __neg__(self) -> Vector:
        """Negation."""
        return Vector(
            data=-self.data,
            grade=self.grade,
            metric=self.metric,
            lot=self.lot,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def __xor__(self, other: Vector | MultiVector) -> Vector | MultiVector:
        """
        Wedge product: u ^ v

        The exterior (wedge) product creates higher-grade Vectors.
        Result grade is grade(u) + grade(v).

        Returns Vector or MultiVector depending on operands.
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Vector):
            from morphis.operations.products import wedge

            return wedge(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _wedge_v_mv

            return _wedge_v_mv(self, other)
        elif isinstance(other, Frame):
            raise TypeError("Wedge product Vector ^ Frame not currently supported")

        return NotImplemented

    def __lshift__(self, other: Vector) -> Vector:
        """
        Left interior product (left contraction): u << v = u ⌋ v

        Contracts all indices of self into other. Result grade is
        grade(other) - grade(self). Returns zero if grade(self) > grade(other).
        """
        if isinstance(other, Vector):
            from morphis.operations.projections import interior_left

            return interior_left(self, other)

        return NotImplemented

    def __rshift__(self, other: Vector) -> Vector:
        """
        Right interior product (right contraction): u >> v = u ⌊ v

        Contracts all indices of other into self. Result grade is
        grade(self) - grade(other). Returns zero if grade(other) > grade(self).
        """
        if isinstance(other, Vector):
            from morphis.operations.projections import interior_right

            return interior_right(self, other)

        return NotImplemented

    def reverse(self) -> Vector:
        """
        Reverse operator.

        Reverses the order of vector factors:
        reverse(u ^ v ^ w) = w ^ v ^ u = (-1)^(k(k-1)/2) * (u ^ v ^ w)

        Returns:
            Reversed Vector
        """
        from morphis.operations.products import reverse

        return reverse(self)

    def rev(self) -> Vector:
        """Short form of reverse()."""
        return self.reverse()

    def __invert__(self) -> Vector:
        """Reverse operator: ~u. Symbol form of reverse()."""
        return self.reverse()

    def inverse(self) -> Vector:
        """
        Multiplicative inverse.

        For unit blades, inverse equals reverse.
        For non-unit: u^(-1) = ~u / (u * ~u)

        Returns:
            Inverse such that u * u.inverse() = 1
        """
        from morphis.operations.products import inverse

        return inverse(self)

    def inv(self) -> Vector:
        """Short form of inverse()."""
        return self.inverse()

    def __pow__(self, exponent: int) -> Vector:
        """
        Power operation.

        Currently supports:
            v**(-1) - multiplicative inverse
            v**(1)  - identity (returns self)
        """
        if exponent == -1:
            return self.inverse()
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only v**(-1) for multiplicative inverse is supported."
            )

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def transform(self, M: MultiVector) -> None:
        """
        Transform this Vector in-place by a motor/versor.

        Performs the sandwich product M * self * ~M and updates self.data.
        This is efficient for animation since no new Vector object is created.

        Args:
            M: MultiVector (motor/versor) to transform by
        """
        from morphis.transforms.actions import transform

        transformed = transform(self, M)
        self.data[...] = transformed.data

    def apply_similarity(
        self,
        S: MultiVector,
        t: "Vector | None" = None,
    ) -> "Vector":
        """
        Apply a similarity transformation to this Vector.

        Computes: transform(self, S) + t

        The similarity versor S (from align_vectors or a rotor) is applied via
        sandwich product. The optional translation t is added after.

        Args:
            S: Similarity versor or rotor (MultiVector with grades {0, 2})
            t: Optional translation vector (grade-1). Added after sandwich product.

        Returns:
            New Vector with the transformation applied.

        Example:
            S = align_vectors(u, v)
            t = Vector([1, 0, 0], grade=1, metric=g)
            w = u.apply_similarity(S, t)  # w = transform(u, S) + t
        """
        from morphis.transforms.actions import transform

        result = transform(self, S)
        if t is not None:
            result = result + t
        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Vector:
        """Create a deep copy of this Vector."""
        return Vector(
            data=self.data.copy(),
            grade=self.grade,
            metric=self.metric,
            lot=self.lot,
        )

    def with_metric(self, metric: Metric) -> Vector:
        """Return a new Vector with the specified metric context."""
        return Vector(
            data=self.data.copy(),
            grade=self.grade,
            metric=metric,
            lot=self.lot,
        )

    def form(self) -> NDArray:
        """
        Compute the quadratic form: v · v.

        This is the metric inner product of the vector with itself.
        Can be negative in non-Euclidean metrics.

        Returns:
            NDArray with shape matching self.lot
        """
        from morphis.operations.norms import form

        return form(self)

    def norm(self) -> NDArray:
        """
        Compute the norm: sqrt(|form(v)|).

        Always returns non-negative values.

        Returns:
            NDArray with shape matching self.lot
        """
        from morphis.operations.norms import norm

        return norm(self)

    def unit(self) -> Vector:
        """
        Return a unit vector (norm = 1).

        For bivectors, this gives the unit bivector needed for rotor construction.
        Handles zero Vectors safely by returning zero.
        """
        from morphis.operations.norms import unit

        return unit(self)

    def conjugate(self) -> Vector:
        """
        Return Vector with complex-conjugated coefficients.

        For real Vectors, returns a copy (conjugation is identity).
        For complex (phasors), conjugates all coefficients.

        Returns:
            Vector with conjugated coefficients
        """
        from morphis.operations.norms import conjugate

        return conjugate(self)

    def conj(self) -> Vector:
        """Short form of conjugate()."""
        return self.conjugate()

    def hodge(self) -> Vector:
        """
        Return Hodge dual.

        Maps grade-k to grade-(dim-k) using the metric.
        The Hodge dual represents the orthogonal complement.
        """
        from morphis.operations.duality import hodge_dual

        return hodge_dual(self)

    # =========================================================================
    # Lot Reductions
    # =========================================================================

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Vector:
        """
        Sum over lot axis/axes, returning a Vector with reduced lot.

        Args:
            axis: Lot axis or axes to sum over. If None, sums over all lot axes.
                  Negative indices count from the end of lot dimensions (not full).

        Returns:
            Vector with summed lot dimensions removed

        Example:
            v = Vector(data, grade=1)  # lot=(M, N, K)
            v.sum(axis=2)              # lot=(M, N), summed over K
            v.sum(axis=(0, 2))         # lot=(N,), summed over M and K
            v.sum()                    # lot=(), summed over all lot axes
        """
        n_lot = len(self.lot)

        if axis is None:
            # Sum over all lot axes
            axes = tuple(range(n_lot))
        elif isinstance(axis, int):
            # Normalize negative index relative to lot dimensions
            if axis < 0:
                axis = n_lot + axis
            if not (0 <= axis < n_lot):
                raise IndexError(f"Lot axis {axis} out of range for {n_lot} lot dimensions")
            axes = (axis,)
        else:
            # Tuple of axes - normalize each
            axes = []
            for ax in axis:
                if ax < 0:
                    ax = n_lot + ax
                if not (0 <= ax < n_lot):
                    raise IndexError(f"Lot axis {ax} out of range for {n_lot} lot dimensions")
                axes.append(ax)
            axes = tuple(axes)

        # Sum over the specified lot axes
        new_data = self.data.sum(axis=axes)

        return Vector(data=new_data, grade=self.grade, metric=self.metric)

    def mean(self, axis: int | tuple[int, ...] | None = None) -> Vector:
        """
        Mean over lot axis/axes, returning a Vector with reduced lot.

        Args:
            axis: Lot axis or axes to average over. If None, averages over all lot axes.
                  Negative indices count from the end of lot dimensions (not full).

        Returns:
            Vector with averaged lot dimensions removed

        Example:
            v = Vector(data, grade=1)  # lot=(M, N, K)
            v.mean(axis=2)             # lot=(M, N), averaged over K
            v.mean(axis=(0, 2))        # lot=(N,), averaged over M and K
            v.mean()                   # lot=(), averaged over all lot axes
        """
        n_lot = len(self.lot)

        if axis is None:
            # Mean over all lot axes
            axes = tuple(range(n_lot))
        elif isinstance(axis, int):
            # Normalize negative index relative to lot dimensions
            if axis < 0:
                axis = n_lot + axis
            if not (0 <= axis < n_lot):
                raise IndexError(f"Lot axis {axis} out of range for {n_lot} lot dimensions")
            axes = (axis,)
        else:
            # Tuple of axes - normalize each
            axes = []
            for ax in axis:
                if ax < 0:
                    ax = n_lot + ax
                if not (0 <= ax < n_lot):
                    raise IndexError(f"Lot axis {ax} out of range for {n_lot} lot dimensions")
                axes.append(ax)
            axes = tuple(axes)

        # Mean over the specified lot axes
        new_data = self.data.mean(axis=axes)

        return Vector(data=new_data, grade=self.grade, metric=self.metric)

    def span(self) -> tuple[Vector, ...]:
        """
        Factor this blade into its constituent grade-1 Vectors.

        For a blade B = v1 ^ v2 ^ ... ^ vk, returns (v1, v2, ..., vk).
        These vectors span the k-dimensional subspace represented by B.

        Note: Factorization is not unique - any k vectors spanning the same
        subspace will work. This returns ONE valid factorization.

        Raises:
            ValueError: If this Vector is not a blade (not factorizable)
        """
        from morphis.operations.factorization import spanning_vectors

        return spanning_vectors(self)

    def __str__(self) -> str:
        from morphis.utils.pretty import format_vector

        return format_vector(self)

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Basis Constructors
# =============================================================================


def basis_vector(index: int, metric: Metric) -> Vector:
    """
    Create the i-th basis vector e_i.

    Args:
        index: Basis index (0-indexed: 0 for e0, 1 for e1, etc.)
        metric: Metric defining the geometric algebra

    Returns:
        Grade-1 Vector with 1 in position index, 0 elsewhere

    Example:
        e1 = basis_vector(0, euclidean_metric(3))
    """
    dim = metric.dim
    data = zeros(dim)
    data[index] = 1.0
    return Vector(data, grade=1, metric=metric)


def basis_vectors(metric: Metric) -> tuple[Vector, ...]:
    """
    Create all dim basis vectors (e0, e1, ..., e_{d-1}).

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Tuple of grade-1 Vectors

    Example:
        e0, e1, e2 = basis_vectors(euclidean_metric(3))
        e01 = e0 ^ e1  # Wedge product creates bivector
    """
    return tuple(basis_vector(k, metric) for k in range(metric.dim))


def basis_element(indices: tuple[int, ...], metric: Metric) -> Vector:
    """
    Create a basis element e_{i0} ^ e_{i1} ^ ... ^ e_{ik}.

    Args:
        indices: Tuple of basis indices (0-indexed)
        metric: Metric defining the geometric algebra

    Returns:
        Vector of grade len(indices)

    Example:
        e01 = basis_element((0, 1), euclidean_metric(3))  # e0 ^ e1
        e012 = basis_element((0, 1, 2), euclidean_metric(3))  # pseudoscalar in 3D
    """
    if not indices:
        raise ValueError("indices must be non-empty; use Vector(1.0, grade=0, metric=m) for scalars")

    result = basis_vector(indices[0], metric)
    for idx in indices[1:]:
        result = result ^ basis_vector(idx, metric)
    return result


def geometric_basis(metric: Metric) -> dict[int, tuple[Vector, ...]]:
    """
    Create complete geometric basis for a metric.

    Returns dictionary mapping grade to tuple of basis elements:
    {0: (1,), 1: (e0, e1, ...), 2: (e01, e02, ...), ..., d: (e0...d,)}

    The number of basis elements at grade k is C(d, k) (binomial coefficient).
    Total number of basis elements is 2^d.

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Dictionary mapping grade to tuple of basis Vectors

    Examples:
        >>> from morphis.elements import geometric_basis, euclidean_metric
        >>> basis = geometric_basis(euclidean_metric(3))
        >>> basis[0]  # Scalar
        (Vector(grade=0, ...),)
        >>> len(basis[1])  # Vectors
        3
        >>> len(basis[2])  # Bivectors
        3
        >>> len(basis[3])  # Trivector (pseudoscalar)
        1

        >>> # Total basis elements: 1 + 3 + 3 + 1 = 8 = 2^3
    """
    from itertools import combinations

    from numpy import ones

    dim = metric.dim
    result = {}

    # Grade 0: scalar basis element (the scalar 1)
    result[0] = (Vector(ones(()), grade=0, metric=metric),)

    # Grades 1 through dim
    for grade in range(1, dim + 1):
        basis_elements = []

        # Generate all combinations of indices for this grade
        for indices in combinations(range(dim), grade):
            # Create basis element for these indices
            elem = basis_element(indices, metric)
            basis_elements.append(elem)

        result[grade] = tuple(basis_elements)

    return result


def pseudoscalar(metric: Metric) -> Vector:
    """
    Create the pseudoscalar (volume element) e_{01...d-1}.

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Grade-d Vector (the unit pseudoscalar)

    Example:
        I = pseudoscalar(euclidean_metric(3))  # e0 ^ e1 ^ e2
    """
    return basis_element(tuple(range(metric.dim)), metric)
