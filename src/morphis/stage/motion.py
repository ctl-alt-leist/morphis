"""
Stage - Motion

Every object in a space can move rigidly in the ways the space allows. Motion
is the group of dials for that: one rate per basis plane, the metric deciding
what each rate is a rate of,

    rotation plane      B² < 0    an angular rate
    Lorentzian plane    B² > 0    a rapidity rate
    null plane e_0m     B² = 0    a velocity (PGA)

and, in a space without a projective direction, where translation is not a
versor, a velocity v moving the object's origin.

Together the rates are a bivector Ω. An object's pose is a versor M and an
offset o, the object drawn as its rest shape carried by M and moved by o. A
step of Δt takes the exact step S = exp(-½ Ω Δt), on either side of the pose:

    M ← S M     extrinsic: Ω is expressed in the planes of the space
    M ← M S     intrinsic: Ω is expressed in the object's own planes

The rates are constant between steps, and steps of one constant Ω compose to
a single one, so the pose after a time is the same however that time is cut
into steps: the flow exp(-½ Ω t) exactly, with no integration error. Ω is
generally not simple, which the general bivector exponential handles.
"""

from typing import Any, Literal, NamedTuple

from numpy import zeros
from pydantic import Field

from morphis.elements import Metric, Vector, basis_vector
from morphis.elements.metric import Label
from morphis.operations.products import geometric, grade_project
from morphis.stage.dial import Dials, DialSpec
from morphis.stage.pose import Pose, unit
from morphis.transforms import rotor


PlaneKind = Literal["rotation", "boost", "translation"]

# What a rate in each kind of plane is a rate of, worded for a front end
QUANTITIES = {"rotation": "angular rate", "boost": "rapidity rate", "translation": "velocity"}

# A plane's square counts as zero below this, as for the null planes of PGA
NULL = 1e-12


class Plane(NamedTuple):
    """A basis plane e_a ∧ e_b, by the labels of its directions, and its kind."""

    first: Label
    second: Label
    kind: PlaneKind

    @property
    def key(self) -> str:
        """The plane as a dial's key: its labels, joined by a comma."""
        return f"{self.first},{self.second}"


class Motion(Dials):
    """
    The rates a rigid motion is driven by, one per basis plane of the space,
    the velocity of the origin where translation is not a versor, and the
    side the step is taken on.

    A rate is addressed by the labels of its plane, in either order, the
    reversed order turning the other way: motion[1, 3] = -motion[3, 1].

    Attributes:
        metric: The space moved in
        rate: The rate in each plane, by the plane's key ("1,3")
        velocity: The velocity of the origin along each direction, by label,
            empty in a projective space
        intrinsic: Whether the rates are expressed in the object's own planes
    """

    metric: Metric = Field(exclude=True, repr=False)
    rate: dict[str, float]
    velocity: dict[str, float]
    intrinsic: bool = False

    def __init__(self, metric: Metric, **data: Any) -> None:
        rates = {plane.key: 0.0 for plane in planes(metric)}
        velocity = {} if metric.is_projective else {str(label): 0.0 for label in metric.labels}
        super().__init__(metric=metric, **{"rate": rates, "velocity": velocity, **data})

    @property
    def planes(self) -> tuple[Plane, ...]:
        return planes(self.metric)

    def __getitem__(self, labels: tuple[Label, Label]) -> float:
        key, sign = self._plane(labels)

        return sign * self.rate[key]

    def __setitem__(self, labels: tuple[Label, Label], rate: float) -> None:
        key, sign = self._plane(labels)
        self.rate[key] = sign * float(rate)

    def _plane(self, labels: tuple[Label, Label]) -> tuple[str, float]:
        """The key of the plane named by two labels, and +1 or -1 for their order."""
        first, second = labels
        forward, backward = f"{first},{second}", f"{second},{first}"
        if forward in self.rate:
            result = forward, 1.0
        elif backward in self.rate:
            result = backward, -1.0
        else:
            raise KeyError(f"no plane e_{first}{second} in this space")

        return result

    # =========================================================================
    # Dials
    # =========================================================================

    def describe(self, prefix: str = "") -> list[DialSpec]:
        """
        Returns the dials in order: a rate per plane, the velocity along each
        direction, then the side the step is taken on.
        """
        rates = [
            DialSpec(path=f"{prefix}rate.{plane.key}", kind="number", rate=True, quantity=QUANTITIES[plane.kind])
            for plane in self.planes
        ]
        velocity = [
            DialSpec(path=f"{prefix}velocity.{label}", kind="number", rate=True, quantity="velocity")
            for label in self.velocity
        ]
        result = [*rates, *velocity, DialSpec(path=f"{prefix}intrinsic", kind="toggle")]

        return result

    # =========================================================================
    # Stepping
    # =========================================================================

    @property
    def generator(self) -> Vector:
        """
        Returns the bivector Ω = Σ ω_ab e_a ∧ e_b of the rates.
        """
        g = self.metric
        result = Vector(zeros((g.dim, g.dim)), grade=2, metric=g)
        for plane in self.planes:
            result = result + self.rate[plane.key] * (basis_vector(plane.first, g) ^ basis_vector(plane.second, g))

        return result

    @property
    def drift(self) -> Vector:
        """
        Returns the velocity v = Σ v_a e_a of the origin.
        """
        g = self.metric
        result = Vector(zeros(g.dim), grade=1, metric=g)
        for label in g.labels if self.velocity else ():
            result = result + self.velocity[str(label)] * basis_vector(label, g)

        return result

    def advance(self, pose: Pose, dt: float) -> Pose:
        """
        Returns the pose a time dt later, the rates held constant.

        The versor takes the exact step S = exp(-½ Ω dt), on the left when
        extrinsic and on the right when intrinsic. The origin moves by v dt,
        with v carried by the pose when intrinsic; that carrying is taken at
        the start of the step, so it is exact when the rates are zero and to
        first order in dt otherwise. In PGA the versor carries translation
        too, and every step is exact.
        """
        S = rotor(self.generator, dt)
        M = pose.versor
        if self.intrinsic:
            versor = M * S
            velocity = (M * self.drift * ~M)[1]
        else:
            versor = S * M
            velocity = self.drift
        result = Pose(versor=unit(versor), offset=pose.offset + dt * velocity)

        return result

    def stop(self) -> None:
        """
        Set every rate and velocity to zero.
        """
        for key in self.rate:
            self.rate[key] = 0.0
        for key in self.velocity:
            self.velocity[key] = 0.0


# =============================================================================
# Planes
# =============================================================================


def planes(metric: Metric) -> tuple[Plane, ...]:
    """
    Returns the basis planes of a space in storage order, each with its kind,
    read off the sign of its square.
    """
    labels = metric.labels
    result = []
    for a, first in enumerate(labels):
        for second in labels[a + 1 :]:
            B = basis_vector(first, metric) ^ basis_vector(second, metric)
            square = float(grade_project(geometric(B, B), 0).data)
            if abs(square) < NULL:
                kind = "translation"
            elif square < 0:
                kind = "rotation"
            else:
                kind = "boost"
            result.append(Plane(first, second, kind))

    return tuple(result)
