"""
Linear Algebra - Blade Specifications

Defines BladeSpec for describing the structure of blades in linear operator contexts.
A BladeSpec captures the grade, collection dimensions, and vector space dimension.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BladeSpec:
    """
    Specification for a blade's structure in a linear operator context.

    Describes how to interpret the axes of a blade tensor:
    - grade: Number of geometric axes (0=scalar, 1=vector, 2=bivector, etc.)
    - collection: Number of leading collection/batch dimensions
    - dim: Dimension of the underlying vector space

    Blade storage convention: (*collection, *geometric) where geometric = (dim,) * grade

    Attributes:
        grade: Grade of the blade (0=scalar, 1=vector, 2=bivector, etc.)
        collection: Number of collection dimensions (batch/sensor/time axes)
        dim: Dimension of the underlying vector space

    Examples:
        >>> # Scalar with collection dim (e.g., N currents)
        >>> spec = BladeSpec(grade=0, collection=1, dim=3)
        >>> spec.geometric_shape
        ()
        >>> spec.total_axes
        1

        >>> # Bivector with collection dim (e.g., M magnetic field measurements)
        >>> spec = BladeSpec(grade=2, collection=1, dim=3)
        >>> spec.geometric_shape
        (3, 3)
        >>> spec.total_axes
        3
    """

    grade: int
    collection: int
    dim: int

    def __post_init__(self):
        """Validate spec parameters."""
        if self.grade < 0:
            raise ValueError(f"grade must be non-negative, got {self.grade}")
        if self.collection < 0:
            raise ValueError(f"collection must be non-negative, got {self.collection}")
        if self.dim < 1:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.grade > self.dim:
            raise ValueError(f"grade {self.grade} cannot exceed dim {self.dim}")

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

    def blade_shape(self, collection_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Compute full blade shape given collection shape.

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


def blade_spec(grade: int, dim: int, collection: int = 1) -> BladeSpec:
    """
    Create a BladeSpec with convenient defaults.

    Args:
        grade: Grade of blade (0=scalar, 1=vector, 2=bivector, etc.)
        dim: Dimension of vector space
        collection: Number of collection dimensions (default 1)

    Returns:
        BladeSpec instance

    Examples:
        >>> # Scalar currents with one collection dimension
        >>> spec = blade_spec(grade=0, dim=3)

        >>> # Bivector fields with one collection dimension
        >>> spec = blade_spec(grade=2, dim=3)

        >>> # Vector without collection dimensions
        >>> spec = blade_spec(grade=1, dim=3, collection=0)
    """
    return BladeSpec(grade=grade, collection=collection, dim=dim)
