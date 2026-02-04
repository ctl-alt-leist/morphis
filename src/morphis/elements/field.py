"""
Field Element - Positions with Values (Skeleton)

Field represents a collection of positions in space, each with an associated
value (scalar, vector, frame, or multivector). This is a skeleton for future
implementation.

Typical uses:
- Scalar fields: temperature, pressure, density
- Vector fields: velocity, electric field, force
- Frame fields: stress tensors, rotation fields
- Multivector fields: electromagnetic field (F = E + IcB)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import ConfigDict, model_validator

from morphis.elements.base import Element
from morphis.elements.vector import Vector


if TYPE_CHECKING:
    from morphis.elements.frame import Frame
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector


class Field(Element):
    """
    A field of values at positions in space.

    SKELETON: This class structure is defined for future implementation.
    Full implementation is deferred.

    Attributes:
        positions: Grade-1 Vector with lot=(N,) representing N sample positions
        values: The field values at each position (Vector, Frame, or MultiVector)

    The values must have lot matching positions.lot, so each position has
    exactly one associated value.

    Examples:
        # Velocity field (3D vectors at 3D positions)
        positions = Vector(sample_points, grade=1, metric=g)  # lot=(100,)
        velocities = Vector(velocity_data, grade=1, metric=g)  # lot=(100,)
        field = Field(positions=positions, values=velocities)

        # Scalar field (scalars at positions)
        temperatures = Vector(temp_data, grade=0, metric=g)  # lot=(100,)
        field = Field(positions=positions, values=temperatures)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    positions: Vector
    values: "Vector | Frame | MultiVector"

    @model_validator(mode="after")
    def _validate_field(self):
        """Validate positions and values are compatible."""
        # Positions must be grade-1 vectors
        if self.positions.grade != 1:
            raise ValueError(f"Positions must be grade-1 Vectors, got grade={self.positions.grade}")

        # Lots must match
        if self.positions.lot != self.values.lot:
            raise ValueError(f"Positions lot {self.positions.lot} != values lot {self.values.lot}")

        # Sync metric and lot from positions
        object.__setattr__(self, "metric", self.positions.metric)
        object.__setattr__(self, "lot", self.positions.lot)

        return self

    @property
    def n_samples(self) -> int:
        """Number of sample positions in the field."""
        return self.positions.lot[0] if self.positions.lot else 1

    def copy(self) -> Self:
        """Create a deep copy of this field."""
        return Field(
            positions=self.positions.copy(),
            values=self.values.copy(),
        )

    def with_metric(self, metric: "Metric") -> Self:
        """Return a new Field with the specified metric context."""
        return Field(
            positions=self.positions.with_metric(metric),
            values=self.values.with_metric(metric),
        )

    def __repr__(self) -> str:
        value_type = type(self.values).__name__
        return f"Field(n_samples={self.n_samples}, value_type={value_type})"
