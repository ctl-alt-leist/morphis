"""
Stage - Operations

An object is moved by its own dials; an operation moves others. A rotation
about a center, a translation, a boost: each is an entry of the tree with
dials of its own, a Motion for what it does, a center it does it about, and a
switch, and it acts on the entries beside it in its group. Since a group
carries its members, everything beneath those entries follows, so an
operation at the root acts on the whole stage, and one inside a group acts on
that group alone. Moving an operation in the tree changes what it reaches.

An operation is on, or applied once. While on, it acts continuously as the
clock plays, each tick taking the exact step of its rates; applied once, its
rates are read as amounts and act for a unit of time, all at once. Operations
in one group act in the order they are listed, since in general they do not
commute.

The map an operation applies is G: y ↦ S (y - c) S̃ + c + v t, with S the
exact step of its rates, c its center, and v its velocity. Extrinsic, it acts
after an entry's drawn pose, in the planes of the space: D ← G ∘ D. Intrinsic,
it acts before it, in the entry's own frame: D ← D ∘ G.

These are global operations: one versor for every point. A local operation, a
versor field such as a gauge transformation, acts on an object's data point
by point and cannot be folded into a pose; it is not designed yet.

In PGA the center is the origin, since translation is among the operation's
own planes there.
"""

from typing import Any

from numpy import zeros
from pydantic import Field

from morphis.elements import Metric, Vector, basis_vector
from morphis.stage.dial import Dials
from morphis.stage.motion import Motion
from morphis.stage.pose import Pose


class Point(Dials):
    """
    A point of the space, by its coordinate along each direction.

    Attributes:
        metric: The space
        at: The coordinate along each direction, by label
    """

    metric: Metric = Field(exclude=True, repr=False)
    at: dict[str, float]

    def __init__(self, metric: Metric, **data: Any) -> None:
        super().__init__(metric=metric, **{"at": {str(label): 0.0 for label in metric.labels}, **data})

    @property
    def vector(self) -> Vector:
        """Returns the point as a vector of the space."""
        g = self.metric
        result = Vector(zeros(g.dim), grade=1, metric=g)
        for label in g.labels:
            result = result + self.at[str(label)] * basis_vector(label, g)

        return result


class Switch(Dials):
    on: bool = False


def operation_dials(metric: Metric) -> dict[str, Dials]:
    """
    Returns an operation's dials: what it does, the center it does it about
    (none in PGA), and whether it is on.
    """
    center = {} if metric.is_projective else {"center": Point(metric)}
    result = {"motion": Motion(metric), **center, "operation": Switch()}

    return result


def operation_map(dials: dict[str, Dials], dt: float) -> Pose:
    """
    Returns the map an operation applies over a time dt: its exact step, taken
    about its center, and its velocity times dt.
    """
    motion = dials["motion"]
    step = motion.advance(Pose.identity(motion.metric), dt)
    result = step.about(dials["center"].vector) if "center" in dials else step

    return result


def act(G: Pose, pose: Pose, pivot: Vector, intrinsic: bool) -> Pose:
    """
    Returns an entry's pose once an operation's map has acted on it: after its
    drawn pose when extrinsic, before it when intrinsic.
    """
    drawn = pose.about(pivot)
    acted = drawn.compose(G) if intrinsic else G.compose(drawn)
    result = acted.rebased(pivot)

    return result
