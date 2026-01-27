"""
Linear Algebra - Vector Specifications

Defines VectorSpec for describing the structure of k-vectors in linear operator contexts.
A VectorSpec captures the grade, collection dimensions, and vector space dimension.
"""

from pydantic import BaseModel, ConfigDict, model_validator


class VectorSpec(BaseModel):
    """
    Specification for a k-vector's structure in a linear operator context.

    Describes how to interpret the axes of a k-vector tensor:
    - grade: Number of geometric axes (0=scalar, 1=vector, 2=bivector, etc.)
    - collection: Number of leading collection/batch dimensions
    - dim: Dimension of the underlying vector space

    Storage convention: (*collection, *geometric) where geometric = (dim,) * grade

    Attributes:
        grade: Grade of the k-vector (0=scalar, 1=vector, 2=bivector, etc.)
        collection: Number of collection dimensions (batch/sensor/time axes)
        dim: Dimension of the underlying vector space

    Examples:
        >>> # Scalar with collection dim (e.g., N currents)
        >>> spec = VectorSpec(grade=0, collection=1, dim=3)
        >>> spec.geometric_shape
        ()
        >>> spec.total_axes
        1

        >>> # Bivector with collection dim (e.g., M magnetic field measurements)
        >>> spec = VectorSpec(grade=2, collection=1, dim=3)
        >>> spec.geometric_shape
        (3, 3)
        >>> spec.total_axes
        3
    """

    model_config = ConfigDict(frozen=True)

    grade: int
    collection: int
    dim: int

    @model_validator(mode="after")
    def _validate_spec(self):
        """Validate spec parameters."""
        if self.grade < 0:
            raise ValueError(f"grade must be non-negative, got {self.grade}")
        if self.collection < 0:
            raise ValueError(f"collection must be non-negative, got {self.collection}")
        if self.dim < 1:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.grade > self.dim:
            raise ValueError(f"grade {self.grade} cannot exceed dim {self.dim}")
        return self

    @property
    def geometric_shape(self) -> tuple[int, ...]:
        """
        Shape of the geometric (trailing) dimensions.

        Returns (dim,) * grade. For scalars (grade=0), returns ().
        """
        return (self.dim,) * self.grade

    @property
    def total_axes(self) -> int:
        """Total number of axes: collection + grade."""
        return self.collection + self.grade

    def vector_shape(self, collection_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Compute full vector shape given collection shape.

        Args:
            collection_shape: Shape of collection dimensions

        Returns:
            Full shape (*collection_shape, *geometric_shape)

        Raises:
            ValueError: If collection_shape length doesn't match collection
        """
        if len(collection_shape) != self.collection:
            raise ValueError(f"collection_shape has {len(collection_shape)} dims, but spec expects {self.collection}")

        return collection_shape + self.geometric_shape


def vector_spec(grade: int, dim: int, collection: int = 1) -> VectorSpec:
    """
    Create a VectorSpec with convenient defaults.

    Args:
        grade: Grade of k-vector (0=scalar, 1=vector, 2=bivector, etc.)
        dim: Dimension of vector space
        collection: Number of collection dimensions (default 1)

    Returns:
        VectorSpec instance

    Examples:
        >>> # Scalar currents with one collection dimension
        >>> spec = vector_spec(grade=0, dim=3)

        >>> # Bivector fields with one collection dimension
        >>> spec = vector_spec(grade=2, dim=3)

        >>> # Vector without collection dimensions
        >>> spec = vector_spec(grade=1, dim=3, collection=0)
    """
    return VectorSpec(grade=grade, collection=collection, dim=dim)
