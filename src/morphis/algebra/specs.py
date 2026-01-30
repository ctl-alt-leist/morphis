"""
Linear Algebra - Vector Specifications

Defines VectorSpec for describing the structure of k-vectors in linear operator contexts.
A VectorSpec captures the grade, lot (collection) shape, and vector space dimension.
"""

import warnings

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class VectorSpec(BaseModel):
    """
    Specification for a k-vector's structure in a linear operator context.

    Describes how to interpret the axes of a k-vector tensor:
    - grade: Number of geometric axes (0=scalar, 1=vector, 2=bivector, etc.)
    - lot: Shape of leading lot (collection/batch) dimensions
    - dim: Dimension of the underlying vector space

    Storage convention: (*lot, *geo) where geo = (dim,) * grade

    Attributes:
        grade: Grade of the k-vector (0=scalar, 1=vector, 2=bivector, etc.)
        lot: Shape of lot dimensions (batch/sensor/time axes)
        dim: Dimension of the underlying vector space

    Examples:
        >>> # Scalar with lot shape (N,) (e.g., N currents)
        >>> spec = VectorSpec(grade=0, lot=(5,), dim=3)
        >>> spec.geo
        ()
        >>> spec.shape
        (5,)

        >>> # Bivector with lot shape (M,) (e.g., M magnetic field measurements)
        >>> spec = VectorSpec(grade=2, lot=(10,), dim=3)
        >>> spec.geo
        (3, 3)
        >>> spec.shape
        (10, 3, 3)
    """

    model_config = ConfigDict(frozen=True)

    grade: int
    lot: tuple[int, ...]
    dim: int

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_collection(cls, data):
        """Convert legacy collection=int to lot=tuple."""
        if isinstance(data, dict) and "collection" in data and "lot" not in data:
            collection = data.pop("collection")
            if isinstance(collection, int):
                # Legacy: collection was a count, not a shape
                # We can't know the actual sizes, so we use placeholder sizes of 1
                # This is for backwards compatibility during transition
                warnings.warn(
                    "VectorSpec(collection=int) is deprecated. Use lot=(size,) tuple instead. "
                    "Using placeholder lot dimensions.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                data["lot"] = (1,) * collection
            else:
                data["lot"] = collection
        return data

    @field_validator("lot", mode="before")
    @classmethod
    def _convert_lot_to_tuple(cls, v):
        """Convert list to tuple for lot."""
        if isinstance(v, list):
            return tuple(v)
        return v

    @model_validator(mode="after")
    def _validate_spec(self):
        """Validate spec parameters."""
        if self.grade < 0:
            raise ValueError(f"grade must be non-negative, got {self.grade}")
        if self.dim < 1:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.grade > self.dim:
            raise ValueError(f"grade {self.grade} cannot exceed dim {self.dim}")
        for i, size in enumerate(self.lot):
            if size < 0:
                raise ValueError(f"lot dimension {i} must be non-negative, got {size}")
        return self

    @property
    def geo(self) -> tuple[int, ...]:
        """
        Shape of the geometric (trailing) dimensions.

        Returns (dim,) * grade. For scalars (grade=0), returns ().
        """
        return (self.dim,) * self.grade

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape: lot + geo."""
        return self.lot + self.geo

    # Backwards compatibility aliases
    @property
    def geometric_shape(self) -> tuple[int, ...]:
        """Alias for geo (backwards compatibility)."""
        return self.geo

    @property
    def collection(self) -> int:
        """Number of lot dimensions (backwards compatibility)."""
        return len(self.lot)

    @property
    def total_axes(self) -> int:
        """Total number of axes: len(lot) + grade."""
        return len(self.lot) + self.grade


def vector_spec(
    grade: int,
    dim: int,
    lot: tuple[int, ...] | None = None,
    collection: int | None = None,
) -> VectorSpec:
    """
    Create a VectorSpec with convenient defaults.

    Args:
        grade: Grade of k-vector (0=scalar, 1=vector, 2=bivector, etc.)
        dim: Dimension of vector space
        lot: Shape of lot dimensions (default empty tuple)
        collection: DEPRECATED. Number of lot dimensions (use lot instead)

    Returns:
        VectorSpec instance

    Examples:
        >>> # Scalar currents with lot shape (5,)
        >>> spec = vector_spec(grade=0, dim=3, lot=(5,))

        >>> # Bivector fields with lot shape (10,)
        >>> spec = vector_spec(grade=2, dim=3, lot=(10,))

        >>> # Vector without lot dimensions
        >>> spec = vector_spec(grade=1, dim=3)
    """
    if lot is None and collection is None:
        lot = ()
    elif lot is None and collection is not None:
        warnings.warn(
            "vector_spec(collection=int) is deprecated. Use lot=(size,) tuple instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        lot = (1,) * collection
    elif lot is not None and collection is not None:
        raise ValueError("Cannot specify both lot and collection")

    return VectorSpec(grade=grade, lot=lot, dim=dim)
