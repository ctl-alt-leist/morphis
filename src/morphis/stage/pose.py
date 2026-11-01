"""
Stage - Poses

A pose is where a moving object is. It is a rigid map of the space, a versor
M and an offset o, acting on directions and on points:

    v ↦ M v M̃            a direction, or a blade
    x ↦ M x M̃ + o        a point

In PGA the versor carries translation too, and the offset stays zero.

Poses compose as maps do, the outer one applied after the inner one:

    (M₁, o₁) ∘ (M₂, o₂) = (M₁ M₂, M₁ o₂ M̃₁ + o₁)

which is how a group's pose carries its members. An object turns about its
pivot c, not about the origin of the space: drawn, it is the pose taken about
the pivot, (M, o + c - M c M̃), so a turn leaves the pivot where it is.
"""

from numpy import sqrt, zeros
from pydantic import BaseModel, ConfigDict

from morphis.elements import Metric, MultiVector, Vector


class Pose(BaseModel):
    """
    A rigid map of the space: directions carried by the versor, points carried
    by the versor and moved by the offset.

    Attributes:
        versor: The rotor, boost, or motor
        offset: The displacement of points, zero in PGA, where the versor
            carries translation too
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    versor: MultiVector
    offset: Vector

    @classmethod
    def identity(cls, metric: Metric) -> "Pose":
        """
        Returns the pose of an object at rest: the unit versor and no offset.
        """
        one = MultiVector(data={0: Vector(1.0, grade=0, metric=metric)}, metric=metric)
        result = cls(versor=one, offset=Vector(zeros(metric.dim), grade=1, metric=metric))

        return result

    def directions(self, v: Vector) -> Vector:
        """
        Returns a direction or blade carried by the versor, M v M̃.
        """
        result = (self.versor * v * ~self.versor)[v.grade]

        return result

    def points(self, x: Vector) -> Vector:
        """
        Returns points carried by the versor and moved by the offset, M x M̃ + o.
        """
        result = self.directions(x) + self.offset

        return result

    def compose(self, inner: "Pose") -> "Pose":
        """
        Returns this pose applied after the inner one.
        """
        result = Pose(versor=unit(self.versor * inner.versor), offset=self.points(inner.offset))

        return result

    def about(self, pivot: Vector) -> "Pose":
        """
        Returns the pose taken about a pivot: the map that turns about the
        pivot by the versor and then moves by the offset.
        """
        result = Pose(versor=self.versor, offset=self.offset + pivot - self.directions(pivot))

        return result

    def rebased(self, pivot: Vector) -> "Pose":
        """
        Returns the pose whose map about the pivot is this map: the inverse of `about`.
        """
        result = Pose(versor=self.versor, offset=self.offset - pivot + self.directions(pivot))

        return result


def unit(versor: MultiVector) -> MultiVector:
    """
    Returns the versor scaled so that M M̃ = 1, against the drift of many products.
    """
    square = float((versor * ~versor)[0].data)
    result = (1.0 / sqrt(abs(square))) * versor

    return result
