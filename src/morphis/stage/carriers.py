"""
Stage - Carriers

How a pose is applied to an object. A carrier takes the object at rest, the
object as drawn, and the pose, and writes the rest shape carried by the pose
into the drawn object in place, so the Scene, which holds the drawn object,
redraws it at the next capture. It decides which parts of the object are
directions, carried by the versor alone, and which are points, moved by the
offset too:

    Vector    a blade, or arrows: directions
    Frame     its vectors: directions
    Surface   its vertices: points
    Mesh      its vertices: points

Where an object is drawn from (an arrow's tail, a frame's corner) is a point,
and is carried by the stage itself. An object with no carrier is drawn as it
is and does not move.

A carrier is registered against a class, and an object is carried by the one
registered for the nearest class it is an instance of.
"""

from collections.abc import Callable
from typing import Any

from morphis.elements import Frame, Surface, Vector
from morphis.manifold import Mesh
from morphis.stage.pose import Pose


Carrier = Callable[[Any, Any, Pose], None]

CARRIERS: dict[type, Carrier] = {}


def carries(kind: type) -> Callable[[Carrier], Carrier]:
    """
    Register a carrier for a class of object.
    """

    def register(carrier: Carrier) -> Carrier:
        CARRIERS[kind] = carrier

        return carrier

    return register


def carrier(obj: Any) -> Carrier | None:
    """
    Returns the carrier for the nearest class the object is an instance of, or
    none.
    """
    result = next((CARRIERS[kind] for kind in type(obj).__mro__ if kind in CARRIERS), None)

    return result


@carries(Vector)
def _vector(rest: Vector, drawn: Vector, pose: Pose) -> None:
    drawn.data[...] = pose.directions(rest).data


@carries(Frame)
def _frame(rest: Frame, drawn: Frame, pose: Pose) -> None:
    drawn.data[...] = rest.transform(pose.versor).data


@carries(Surface)
def _surface(rest: Surface, drawn: Surface, pose: Pose) -> None:
    drawn.vertices.data[...] = pose.points(rest.vertices).data


@carries(Mesh)
def _mesh(rest: Mesh, drawn: Mesh, pose: Pose) -> None:
    drawn.vertices.data[...] = pose.points(rest.vertices).data
