"""
Geometric Algebra - Frame

An ordered collection of vectors in d-dimensional space. A Frame preserves
the specific choice of spanning vectors, unlike a Vector which encodes only
the subspace.

Every Frame requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import stack, where
from pydantic import ConfigDict, model_validator

from morphis.config import TOLERANCE
from morphis.elements.base import GradedElement
from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


class Frame(GradedElement):
    """
    An ordered collection of grade-1 vectors in d-dimensional space.

    A frame F = {e1, e2, ..., ek} represents vectors that span a subspace.
    Unlike a k-vector (which encodes only the subspace), a frame preserves the
    specific choice of spanning vectors.

    Storage shape is (*lot, span, dim) where:
    - lot: shape of lot (collection) dimensions (for batch operations)
    - span: number of vectors in the frame
    - dim: dimension of each vector

    Key distinction from k-Vector:
    - k-Vector: holistic object, transforms as b' = M b ~M
    - Frame: collection of grade-1 vectors, transforms component-wise

    Attributes:
        data: The underlying array of frame vectors (inherited)
        grade: Always 1 for frames (vectors) (inherited)
        metric: The complete geometric context (inherited)
        lot: Shape of the lot (collection) dimensions (inherited)
        span: Number of vectors in the frame

    Examples:
        >>> from morphis.elements.metric import euclidean_metric
        >>> m = euclidean_metric(3)
        >>> F = Frame([[1, 0, 0], [0, 1, 0]], metric=m)
        >>> F.span
        2

        >>> # Or from Vectors:
        >>> e1, e2 = basis_vectors(m)[:2]
        >>> F = Frame(e1, e2)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    grade: int = 1  # Frames contain grade-1 elements (vectors)
    span: int | None = None  # Number of vectors, inferred from data.shape[-2]

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(self, *args, **kwargs):
        """
        Create a Frame from array data or Vectors.

        Supports two forms:
            Frame(v1, v2, v3)           # positional grade-1 Vectors
            Frame(data, metric=m)       # array data with keyword args

        When using positional Vectors:
            - All vectors must be grade-1
            - All vectors must have compatible metrics
        """
        from morphis.elements.vector import Vector

        # Check if positional arguments are Vectors
        if args and all(isinstance(v, Vector) for v in args):
            # Validate all are grade-1
            for i, v in enumerate(args):
                if v.grade != 1:
                    raise ValueError(f"Frame requires grade-1 vectors, vector {i} has grade {v.grade}")

            # Merge metrics and stack data
            metric = Metric.merge(*(v.metric for v in args))
            data = stack([v.data for v in args], axis=0)

            kwargs["data"] = data
            kwargs["span"] = len(args)
            kwargs["metric"] = metric
            kwargs["collection"] = ()
        elif args:
            # Single positional arg (data array)
            if len(args) == 1:
                kwargs["data"] = args[0]
            else:
                raise TypeError(
                    "Frame positional arguments must be grade-1 Vectors. Use Frame(data=arr, metric=m) for array form."
                )

        super().__init__(**kwargs)

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def _infer_and_validate(self):
        """Infer span and lot if not provided, then validate shape consistency."""
        dim = self.metric.dim

        # Frame data must have at least 2 dimensions: (span, dim)
        if self.data.ndim < 2:
            raise ValueError(f"Frame data must have at least 2 dimensions, got {self.data.ndim}")

        # Infer span from second-to-last axis
        if self.span is None:
            object.__setattr__(self, "span", self.data.shape[-2])

        # Infer lot from data shape if not provided
        if self.lot is None:
            lot_ndim = self.data.ndim - 2  # everything except (span, dim)
            object.__setattr__(self, "lot", self.data.shape[:lot_ndim])
        else:
            # Lot was explicitly provided - validate it matches actual shape
            expected_lot = self.data.shape[: len(self.lot)]
            if expected_lot != self.lot:
                raise ValueError(
                    f"Explicit lot {self.lot} does not match "
                    f"actual data shape {self.data.shape}. "
                    f"Expected lot {expected_lot} from shape."
                )

        # Validate: len(lot) + 2 == ndim
        expected_ndim = len(self.lot) + 2
        if self.data.ndim != expected_ndim:
            raise ValueError(f"Frame with lot={self.lot} expects {expected_ndim} dimensions, got {self.data.ndim}")

        # Validate shape matches span and dim
        if self.data.shape[-1] != dim:
            raise ValueError(f"Last axis has size {self.data.shape[-1]}, expected dim={dim}")
        if self.data.shape[-2] != self.span:
            raise ValueError(f"Second-to-last axis has size {self.data.shape[-2]}, expected span={self.span}")

        return self

    # =========================================================================
    # Properties
    # =========================================================================
    # (dim, shape, collection inherited from GradedElement)

    # =========================================================================
    # Vector Access
    # =========================================================================

    def vector(self, i: int) -> Vector:
        """
        Extract the i-th vector as a grade-1 Vector.

        Args:
            i: Index of vector (0-indexed)

        Returns:
            Grade-1 Vector containing the i-th vector
        """
        from morphis.elements.vector import Vector

        return Vector(
            data=self.data[..., i, :].copy(),
            grade=1,
            metric=self.metric,
            lot=self.lot,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def as_vector(self) -> Vector:
        """
        View frame vectors as a batch of grade-1 vectors.

        Returns Vector with collection = (*self.lot, span), treating
        the vectors as a batch for vectorized geometric operations.
        """
        from morphis.elements.vector import Vector

        return Vector(
            data=self.data,
            grade=1,
            metric=self.metric,
            lot=self.lot + (self.span,),
        )

    def __mul__(self, other) -> MultiVector | Frame:
        """
        Multiplication: F * x

        - F * scalar → Frame (scalar multiplication)
        - F * Vector/MultiVector → MultiVector (geometric product)

        For sandwich products: (M * F * ~M)[1] gives transformed vectors.
        """
        from morphis.elements.multivector import MultiVector
        from morphis.elements.vector import Vector
        from morphis.operations.operator import Operator

        # Scalar multiplication (numeric)
        if isinstance(other, (int, float, complex)):
            return Frame(data=self.data * other, metric=self.metric)

        # Scalar multiplication (grade-0 Vector)
        if isinstance(other, Vector) and other.grade == 0:
            return Frame(data=self.data * other.data, metric=self.metric)

        # Geometric product with Vector/MultiVector
        if isinstance(other, (Vector, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(self.as_vector(), other)

        # Operator on right side not supported
        if isinstance(other, Operator):
            raise TypeError("Frame * Operator not currently supported (use L * f)")

        return NotImplemented

    def __rmul__(self, other) -> MultiVector | Frame:
        """
        Reverse multiplication: x * F

        - scalar * F → Frame (scalar multiplication, commutative)
        - Vector/MultiVector * F → MultiVector (geometric product)
        """
        from morphis.elements.multivector import MultiVector
        from morphis.elements.vector import Vector

        # Scalar multiplication (numeric, commutative)
        if isinstance(other, (int, float, complex)):
            return Frame(data=other * self.data, metric=self.metric)

        # Scalar multiplication (grade-0 Vector, commutative)
        if isinstance(other, Vector) and other.grade == 0:
            return Frame(data=other.data * self.data, metric=self.metric)

        # Geometric product with Vector/MultiVector
        if isinstance(other, (Vector, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(other, self.as_vector())

        return NotImplemented

    def __xor__(self, other):
        """Wedge product: F ^ x (not currently supported)."""
        raise TypeError(f"Wedge product Frame ^ {type(other).__name__} not currently supported")

    def __rxor__(self, other):
        """Wedge product (reversed): x ^ F (not currently supported)."""
        raise TypeError(f"Wedge product {type(other).__name__} ^ Frame not currently supported")

    def __invert__(self) -> Frame:
        """
        Reverse operator: ~F

        Frames are invariant under reversal (grade-1 vectors unchanged).
        """
        return self

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def transform(self, M: MultiVector) -> Frame:
        """
        Transform this frame by a motor/versor via sandwich product.

        Computes M * F * ~M, extracting grade-1 components.

        Args:
            M: MultiVector (motor/versor) to transform by

        Returns:
            New Frame with transformed vectors
        """
        result = (M * self * ~M)[1]
        return Frame(data=result.data.copy(), metric=self.metric)

    def transform_inplace(self, M: MultiVector) -> None:
        """
        Transform this frame in-place by a motor/versor.

        Args:
            M: MultiVector (motor/versor) to transform by
        """
        self.data[...] = (M * self * ~M)[1].data

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_vector(self) -> Vector:
        """
        Convert frame to k-vector by wedging all vectors.

        Returns the k-vector: e1 ^ e2 ^ ... ^ e_span
        """
        from morphis.operations.products import wedge

        # Wedge all vectors together
        result = self.vector(0)
        for i in range(1, self.span):
            result = wedge(result, self.vector(i))

        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def normalize(self) -> Frame:
        """
        Return a new frame with each vector normalized to unit length.

        This creates an orthonormal frame if the vectors were originally
        orthogonal.

        Returns:
            New Frame with unit-length vectors
        """
        from numpy.linalg import norm as np_norm

        new_data = self.data.copy()
        for i in range(self.span):
            vec = new_data[..., i, :]
            n = np_norm(vec, axis=-1, keepdims=True)
            # Safe division - avoid divide by zero
            n = where(n > TOLERANCE, n, 1.0)
            new_data[..., i, :] = vec / n

        return Frame(
            data=new_data,
            span=self.span,
            metric=self.metric,
            lot=self.lot,
        )

    def copy(self) -> Frame:
        """Create a deep copy of this frame."""
        return Frame(
            data=self.data.copy(),
            span=self.span,
            metric=self.metric,
            lot=self.lot,
        )

    def with_metric(self, metric: Metric) -> Frame:
        """Return a new Frame with the specified metric context."""
        return Frame(
            data=self.data.copy(),
            span=self.span,
            metric=metric,
            lot=self.lot,
        )

    def __str__(self) -> str:
        from morphis.utils.pretty import format_frame

        return format_frame(self)

    def __repr__(self) -> str:
        return self.__str__()
