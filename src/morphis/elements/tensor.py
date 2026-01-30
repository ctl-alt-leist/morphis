"""
Geometric Algebra - Tensor Base Class

A Tensor represents a general (p,q)-tensor with p contravariant (upper) indices
and q covariant (lower) indices.

Storage shape is (*lot, *contravariant_dims, *covariant_dims) where:
- lot: batch/collection dimensions
- contravariant_dims: (dim,) * contravariant
- covariant_dims: (dim,) * covariant

Vector inherits from Tensor with covariant=0.
"""

from __future__ import annotations

from typing import Self

from numpy import asarray
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_validator

from morphis.elements.base import Element
from morphis.elements.metric import Metric


class Tensor(Element):
    """
    A general (p,q)-tensor in geometric algebra.

    Storage shape is (*lot, *contravariant_dims, *covariant_dims) where
    contravariant_dims and covariant_dims are each (dim,) repeated the appropriate
    number of times.

    Attributes:
        data: The underlying array of tensor components
        contravariant: Number of contravariant (upper) indices
        covariant: Number of covariant (lower) indices
        metric: The complete geometric context (inherited)
        lot: Shape of the lot (collection) dimensions (inherited)

    Examples:
        >>> from morphis.elements.metric import euclidean_metric
        >>> m = euclidean_metric(3)
        >>> # A (1,0)-tensor (vector) in 3D
        >>> t = Tensor(data=[1, 0, 0], contravariant=1, covariant=0, metric=m)
        >>> t.geo
        (3,)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    data: NDArray
    contravariant: int = 0
    covariant: int = 0

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

    @field_validator("contravariant")
    @classmethod
    def _validate_contravariant(cls, v):
        if v < 0:
            raise ValueError(f"contravariant must be non-negative, got {v}")
        return v

    @field_validator("covariant")
    @classmethod
    def _validate_covariant(cls, v):
        if v < 0:
            raise ValueError(f"covariant must be non-negative, got {v}")
        return v

    @model_validator(mode="after")
    def _infer_and_validate(self):
        """Infer metric and lot if not provided, then validate shape."""
        rank = self.contravariant + self.covariant
        actual_ndim = self.data.ndim

        # Infer metric from data shape if not provided
        if self.metric is None:
            if rank == 0:
                # Scalar: ambiguous dimension, default to 3D
                dim = 3
            else:
                # Use last axis size as dimension
                dim = self.data.shape[-1]

            from morphis.elements.metric import euclidean_metric

            object.__setattr__(self, "metric", euclidean_metric(dim))

        dim = self.metric.dim

        # Infer lot from data shape if not provided
        if self.lot is None:
            lot_ndim = actual_ndim - rank
            if lot_ndim < 0:
                raise ValueError(
                    f"Array has {actual_ndim} dimensions but tensor rank "
                    f"({self.contravariant}, {self.covariant}) requires at least {rank} geometric axes"
                )
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

        # Validate: len(lot) + rank == ndim
        if len(self.lot) + rank != actual_ndim:
            raise ValueError(f"len(lot)={len(self.lot)} + rank={rank} != ndim={actual_ndim}")

        # Validate: geometric axes match dim
        if rank > 0:
            for k in range(1, rank + 1):
                if self.data.shape[-k] != dim:
                    raise ValueError(f"Geometric axis {-k} has size {self.data.shape[-k]}, expected {dim}")

        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def rank(self) -> tuple[int, int]:
        """Tensor rank as (contravariant, covariant) tuple."""
        return (self.contravariant, self.covariant)

    @property
    def total_rank(self) -> int:
        """Total rank (contravariant + covariant)."""
        return self.contravariant + self.covariant

    @property
    def geo(self) -> tuple[int, ...]:
        """Shape of the trailing geometric dimensions."""
        return self.data.shape[len(self.lot) :]

    @property
    def contravariant_shape(self) -> tuple[int, ...]:
        """Shape of the contravariant (upper index) dimensions."""
        start = len(self.lot)
        end = start + self.contravariant
        return self.data.shape[start:end]

    @property
    def covariant_shape(self) -> tuple[int, ...]:
        """Shape of the covariant (lower index) dimensions."""
        if self.covariant == 0:
            return ()
        start = len(self.lot) + self.contravariant
        return self.data.shape[start:]

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        return self.data.ndim

    # Backwards compatibility aliases
    @property
    def collection(self) -> tuple[int, ...]:
        """Alias for lot (backwards compatibility)."""
        return self.lot

    @property
    def geometric_shape(self) -> tuple[int, ...]:
        """Alias for geo (backwards compatibility)."""
        return self.geo

    # =========================================================================
    # NumPy Interface
    # =========================================================================

    def __getitem__(self, index):
        """Index into the tensor's data array."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Set values in the underlying array."""
        self.data[index] = value

    def __array__(self, dtype=None):
        """Allow np.asarray(tensor) to work."""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Self:
        """Create a deep copy of this tensor."""
        return Tensor(
            data=self.data.copy(),
            contravariant=self.contravariant,
            covariant=self.covariant,
            metric=self.metric,
            lot=self.lot,
        )

    def with_metric(self, metric: Metric) -> Self:
        """Return a new tensor with the specified metric context."""
        return Tensor(
            data=self.data.copy(),
            contravariant=self.contravariant,
            covariant=self.covariant,
            metric=metric,
            lot=self.lot,
        )

    def __repr__(self) -> str:
        return (
            f"Tensor(rank=({self.contravariant},{self.covariant}), "
            f"dim={self.dim}, collection={self.collection}, shape={self.shape})"
        )
