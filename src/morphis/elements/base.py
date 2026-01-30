"""
Geometric Algebra - Element Base Classes

Base classes for all geometric algebra objects. Every GA element has a metric
that defines its complete geometric context.

Hierarchy:
    Element (metric, lot)
    ├── Tensor (+ data: NDArray, contravariant: int, covariant: int)
    │   └── Vector (antisymmetric, covariant=0)
    ├── GradedElement (+ data: NDArray, grade: int)
    │   └── Frame (+ span: int)
    └── CompositeElement (+ data: dict[int, Vector])
        └── MultiVector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from numpy import asarray
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


class Element(BaseModel):
    """
    Base class for all geometric algebra elements.

    Every GA element has a metric that defines its geometric context.
    The metric provides:
    - The inner product structure (metric tensor g_{ab})
    - The signature type (EUCLIDEAN, LORENTZIAN, DEGENERATE)
    - The structure type (FLAT, PROJECTIVE, CONFORMAL, ROUND)

    If no metric is provided, a Euclidean metric is inferred from the data shape.

    Attributes:
        metric: The complete geometric context (optional, defaults to Euclidean)
        lot: Shape of the lot (collection) dimensions
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    metric: Metric | None = None
    lot: tuple[int, ...] | None = None

    # Backwards compatibility alias
    @property
    def collection(self) -> tuple[int, ...] | None:
        """Alias for lot (backwards compatibility)."""
        return self.lot

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.metric.dim


class GradedElement(Element):
    """
    Base class for elements with a single grade and array data.

    GradedElements store their geometric content in a NumPy array with shape
    (*lot, *geo) where geo shape depends on grade.

    Subclasses: Frame

    Note: Vector inherits from Tensor (not GradedElement) to properly model
    the tensor structure of antisymmetric k-vectors.

    Attributes:
        data: The underlying array of element data
        grade: The grade (0=scalar, 1=vector, 2=bivector, etc.)
    """

    data: NDArray
    grade: int

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("data", mode="before")
    @classmethod
    def _convert_to_array(cls, v):
        """Convert lists, tuples, or arrays to numpy ndarray.

        Preserves complex dtype for phasor support. Integer and float
        arrays are coerced to float64 for consistency.
        """
        arr = asarray(v)
        # Coerce int/float to float64, preserve complex
        if arr.dtype.kind in ("i", "u", "f"):
            return arr.astype(float)
        return arr

    @model_validator(mode="after")
    def _infer_metric_if_needed(self):
        """
        Infer Euclidean metric from data shape if not provided.

        For grade > 0: dimension is inferred from the last axis of data.
        For grade = 0 (scalars): defaults to 3D Euclidean.
        """
        if self.metric is None:
            if self.grade == 0:
                # Scalar: ambiguous dimension, default to 3D
                dim = 3
            else:
                # Use last axis size as dimension
                dim = self.data.shape[-1]

            from morphis.elements.metric import euclidean_metric

            object.__setattr__(self, "metric", euclidean_metric(dim))

        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        return self.data.ndim

    # =========================================================================
    # NumPy Interface
    # =========================================================================

    def __getitem__(self, index):
        """Index into the element's data array."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Set values in the underlying array."""
        self.data[index] = value

    def __array__(self, dtype=None):
        """Allow np.asarray(element) to work."""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Self:
        """Create a deep copy of this element."""
        # This will be overridden by subclasses for proper construction
        raise NotImplementedError("Subclasses must implement copy()")

    def with_metric(self, metric: Metric) -> Self:
        """Return a new element with the specified metric context."""
        # This will be overridden by subclasses for proper construction
        raise NotImplementedError("Subclasses must implement with_metric()")


class CompositeElement(Element):
    """
    Base class for elements composed of multiple grades.

    CompositeElements store components as a dictionary mapping grade to
    GradedElement (sparse representation).

    Subclasses: MultiVector

    Attributes:
        data: Dictionary mapping grade to component GradedElement
    """

    data: dict[int, "Vector"]

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def grades(self) -> list[int]:
        """List of grades with nonzero components."""
        return sorted(self.data.keys())

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Self:
        """Create a deep copy of this element."""
        # This will be overridden by subclasses for proper construction
        raise NotImplementedError("Subclasses must implement copy()")

    def with_metric(self, metric: Metric) -> Self:
        """Return a new element with the specified metric context."""
        # This will be overridden by subclasses for proper construction
        raise NotImplementedError("Subclasses must implement with_metric()")
