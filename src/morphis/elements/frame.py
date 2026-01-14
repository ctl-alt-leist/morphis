"""
Geometric Algebra - Frame

An ordered collection of k vectors in d-dimensional space. A Frame preserves
the specific choice of spanning vectors, unlike a Blade which encodes only
the subspace.

Every Frame requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import asarray, stack
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_validator

from morphis.elements.base import GAModel
from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.blade import Blade
    from morphis.elements.multivector import MultiVector


class Frame(GAModel):
    """
    An ordered collection of k vectors in d-dimensional space.

    A frame F = {e1, e2, ..., ek} represents k basis vectors that span a
    k-dimensional subspace. Unlike a blade (which encodes only the subspace),
    a frame preserves the specific choice of spanning vectors.

    Storage shape is (*collection, k, d) where:
    - collection: shape of collection dimensions (for batch operations)
    - k: number of vectors in the frame
    - d: dimension of each vector

    Key distinction from Blade:
    - Blade: holistic object, transforms as b' = M b ~M
    - Frame: collection of vectors, transforms component-wise

    Attributes:
        data: The underlying array of frame vectors
        k: Number of vectors (inferred if not provided)
        metric: The complete geometric context (required)
        collection: Shape of the collection dimensions

    Examples:
        >>> from morphis.elements.metric import euclidean
        >>> m = euclidean(3)
        >>> F = Frame([[1, 0, 0], [0, 1, 0]], metric=m)
        >>> F.k
        2
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    data: NDArray
    k: int | None = None  # Number of vectors, inferred from data.shape[-2]
    metric: Metric
    collection: tuple[int, ...] | None = None  # Inferred from data shape if not provided

    # Prevent numpy from intercepting arithmetic
    __array_ufunc__ = None

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(self, data=None, /, **kwargs):
        """Allow positional argument for data: Frame(arr, metric=m)."""
        if data is not None:
            kwargs["data"] = data
        super().__init__(**kwargs)

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("data", mode="before")
    @classmethod
    def _convert_to_array(cls, v):
        return asarray(v, dtype=float)

    @model_validator(mode="after")
    def _infer_and_validate(self):
        """Infer k and collection if not provided, then validate shape consistency."""
        dim = self.metric.dim

        # Frame data must have at least 2 dimensions: (k, d)
        if self.data.ndim < 2:
            raise ValueError(f"Frame data must have at least 2 dimensions, got {self.data.ndim}")

        # Infer k from second-to-last axis
        if self.k is None:
            object.__setattr__(self, "k", self.data.shape[-2])

        # Infer collection from data shape if not provided
        if self.collection is None:
            collection_ndim = self.data.ndim - 2  # everything except (k, d)
            object.__setattr__(self, "collection", self.data.shape[:collection_ndim])

        # Validate: len(collection) + 2 == ndim
        expected_ndim = len(self.collection) + 2
        if self.data.ndim != expected_ndim:
            raise ValueError(
                f"Frame with collection={self.collection} expects {expected_ndim} dimensions, got {self.data.ndim}"
            )

        # Validate shape matches k and dim
        if self.data.shape[-1] != dim:
            raise ValueError(f"Last axis has size {self.data.shape[-1]}, expected dim={dim}")
        if self.data.shape[-2] != self.k:
            raise ValueError(f"Second-to-last axis has size {self.data.shape[-2]}, expected k={self.k}")

        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def dim(self) -> int:
        """Dimension of each vector in the frame."""
        return self.metric.dim

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape of the underlying array."""
        return self.data.shape

    @property
    def collection_shape(self) -> tuple[int, ...]:
        """Shape of the leading collection dimensions."""
        return self.collection

    @property
    def grade(self) -> int:
        """Grade of each vector in the frame (always 1)."""
        return 1

    # =========================================================================
    # NumPy Interface
    # =========================================================================

    def __getitem__(self, index):
        """Index into the frame's data array."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Set values in the underlying array."""
        self.data[index] = value

    def __array__(self, dtype=None):
        """Allow np.asarray(frame) to work."""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    # =========================================================================
    # Vector Access
    # =========================================================================

    def vector(self, i: int) -> Blade:
        """
        Extract the i-th vector as a grade-1 Blade.

        Args:
            i: Index of vector (0-indexed)

        Returns:
            Grade-1 Blade containing the i-th vector
        """
        from morphis.elements.blade import Blade

        return Blade(
            data=self.data[..., i, :].copy(),
            grade=1,
            metric=self.metric,
            collection=self.collection,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def _as_blade(self) -> Blade:
        """
        View frame vectors as a batch of grade-1 blades (internal helper).

        Returns Blade with collection = (*self.collection, k), treating
        the k vectors as a batch for vectorized geometric operations.
        """
        from morphis.elements.blade import Blade

        return Blade(
            data=self.data,
            grade=1,
            metric=self.metric,
            collection=self.collection + (self.k,),
        )

    def __matmul__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Geometric product: F @ M

        Treats frame as batch of k vectors and applies geometric product.
        Result is MultiVector with collection (*self.collection, k).

        For sandwich products: (M @ F @ ~M)[1] gives transformed vectors.
        """
        from morphis.elements.blade import Blade
        from morphis.elements.multivector import MultiVector

        if isinstance(other, (Blade, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(self._as_blade(), other)

        return NotImplemented

    def __rmatmul__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Geometric product (reversed): M @ F

        Treats frame as batch of k vectors and applies geometric product.
        Result is MultiVector with collection (*self.collection, k).

        For sandwich products: (M @ F @ ~M)[1] gives transformed vectors.
        """
        from morphis.elements.blade import Blade
        from morphis.elements.multivector import MultiVector

        if isinstance(other, (Blade, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(other, self._as_blade())

        return NotImplemented

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

        Computes M @ F @ ~M, extracting grade-1 components.

        Args:
            M: MultiVector (motor/versor) to transform by

        Returns:
            New Frame with transformed vectors
        """
        result = (M @ self @ ~M)[1]
        return Frame(data=result.data.copy(), metric=self.metric)

    def transform_inplace(self, M: MultiVector) -> None:
        """
        Transform this frame in-place by a motor/versor.

        Args:
            M: MultiVector (motor/versor) to transform by
        """
        self.data[...] = (M @ self @ ~M)[1].data

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_blade(self) -> Blade:
        """
        Convert frame to blade by wedging all vectors.

        Returns the k-blade: e1 ^ e2 ^ ... ^ ek
        """
        from morphis.operations.products import wedge

        # Wedge all vectors together
        result = self.vector(0)
        for i in range(1, self.k):
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
        from numpy import where
        from numpy.linalg import norm as np_norm

        new_data = self.data.copy()
        for i in range(self.k):
            vec = new_data[..., i, :]
            n = np_norm(vec, axis=-1, keepdims=True)
            # Safe division - avoid divide by zero
            n = where(n > 1e-12, n, 1.0)
            new_data[..., i, :] = vec / n

        return Frame(
            data=new_data,
            k=self.k,
            metric=self.metric,
            collection=self.collection,
        )

    def copy(self) -> Frame:
        """Create a deep copy of this frame."""
        return Frame(
            data=self.data.copy(),
            k=self.k,
            metric=self.metric,
            collection=self.collection,
        )

    def with_metric(self, metric: Metric) -> Frame:
        """Return a new Frame with the specified metric context."""
        return Frame(
            data=self.data.copy(),
            k=self.k,
            metric=metric,
            collection=self.collection,
        )

    def __repr__(self) -> str:
        return f"Frame(k={self.k}, dim={self.dim}, collection={self.collection}, shape={self.shape})"


# =============================================================================
# Constructor Functions
# =============================================================================


def frame_from_vectors(*vectors: Blade) -> Frame:
    """
    Create a Frame from a collection of grade-1 Blades (vectors).

    All vectors must have the same metric. The resulting Frame has shape (k, d)
    where k is the number of vectors and d is the dimension.

    Args:
        *vectors: Grade-1 Blades to include in the frame

    Returns:
        Frame containing all vectors

    Example:
        e1, e2, e3 = basis_vectors(metric=euclidean(3))
        E = frame_from_vectors(e1, e2, e3)  # Frame with shape (3, 3)
    """
    from morphis.elements.blade import Blade

    if not vectors:
        raise ValueError("At least one vector required")

    # Validate all are grade-1
    for i, v in enumerate(vectors):
        if not isinstance(v, Blade):
            raise TypeError(f"Vector {i} is not a Blade")
        if v.grade != 1:
            raise ValueError(f"Vector {i} has grade {v.grade}, expected 1")

    # Merge all metrics (raises if incompatible)
    metric = Metric.merge(*(v.metric for v in vectors))

    # Stack vector data: each is shape (d,), result is (k, d)
    data = stack([v.data for v in vectors], axis=0)

    return Frame(data=data, k=len(vectors), metric=metric, collection=())
