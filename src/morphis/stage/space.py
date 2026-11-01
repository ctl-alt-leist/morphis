"""
Stage - The Space's Dials

The dials of the whole picture, held by the space at the root of the tree:

    clock     how fast stage time runs against the wall clock
    view      the projective view of a space of more than three dimensions:
              the eye κ, the shear of the dropped directions, the turn of the
              screen in each plane
    camera    where the renderer's camera stands and what it looks at, in the
              three dimensions of the picture

Selecting the space puts these in the inspector, so the space is worked as
any object is.
"""

from typing import Any

from pydantic import Field

from morphis.elements import Metric
from morphis.manifold.domain import Circle, Interval
from morphis.stage.dial import Dial, Dials
from morphis.stage.motion import planes
from morphis.visuals.projective import Projective


AXES = ("x", "y", "z")

# Where the camera starts: looking at the origin from above one corner
CAMERA_POSITION = (4.9, -4.2, 3.3)


class Clock(Dials):
    speed: float = Dial(1.0, Interval(0.0, 10.0), description="Stage seconds per second of the wall clock")


class View(Dials):
    """
    The projective view, by its coordinates on the family of views.

    Attributes:
        metric: The space viewed
        eye: The inverse distance κ of the eye along the dropped direction
        shear: The picture vector a unit step along the dropped direction draws to
        turn: The screen's turn in each plane, an angle by the plane's key
    """

    metric: Metric = Field(exclude=True, repr=False)
    eye: float = Dial(0.0, Interval(-0.9, 0.9))
    shear: dict[str, float]
    turn: dict[str, float] = Dial({}, Circle())

    def __init__(self, metric: Metric, **data: Any) -> None:
        turn = {plane.key: 0.0 for plane in planes(metric) if plane.kind == "rotation"}
        super().__init__(metric=metric, **{"shear": dict.fromkeys(AXES, 0.0), "turn": turn, **data})

    @property
    def projective(self) -> Projective:
        """Returns the view these dials name."""
        turn = {
            (plane.first, plane.second): self.turn[plane.key] for plane in planes(self.metric) if plane.key in self.turn
        }
        result = Projective(eye=self.eye, shear=tuple(self.shear[axis] for axis in AXES), turn=turn)

        return result


class Camera(Dials):
    """
    Attributes:
        position: Where the camera stands, by picture axis
        focus: The point it looks at, by picture axis
    """

    position: dict[str, float] = Field(default_factory=lambda: dict(zip(AXES, CAMERA_POSITION, strict=True)))
    focus: dict[str, float] = Field(default_factory=lambda: dict.fromkeys(AXES, 0.0))

    @property
    def placement(self) -> dict[str, tuple[float, float, float]]:
        """Returns the camera's position and focal point, as the Scene takes them."""
        result = {
            "position": tuple(self.position[axis] for axis in AXES),
            "focal_point": tuple(self.focus[axis] for axis in AXES),
        }

        return result


def space_dials(metric: Metric) -> dict[str, Dials]:
    """
    Returns the space's dials: the clock, the view, and the camera.
    """
    result = {"clock": Clock(), "view": View(metric), "camera": Camera()}

    return result
