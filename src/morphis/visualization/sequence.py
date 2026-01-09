"""
Animation Timeline and Actions

A clean, elegant system for defining animation sequences. The timeline is a
list of Actions that specify what happens to objects over time. Actions
include drawing, hiding, and transforming blades using Motors.

Key design principles:
1. Actions contain GA operations (Motors), not VTK transforms
2. The canvas reads actions and applies them - separation of concerns
3. Smoothing functions control how transformations are distributed over time

Usage:
    timeline = Timeline()
    timeline.draw(e1, duration=1.0)
    timeline.draw(e2, duration=1.0)
    timeline.draw(e12, duration=1.0)
    timeline.rotate(e123, axis=[1,1,1], angle=2*pi, duration=4.0)

    canvas = AnimatedCanvas()
    canvas.play(timeline)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from numpy import array
from numpy.typing import NDArray

from morphis.core.smoothing import Smoother, get_smoother, smooth_linear
from morphis.ga.model import Blade
from morphis.visualization.theme import Color


DrawMode = Literal["fade_in", "scale_in"]
HideMode = Literal["fade_out", "scale_out"]


@dataclass
class Action(ABC):
    """
    Base class for timeline actions.

    Each action has a duration and operates on one or more blades.
    Actions know how to compute their state at any point in their duration.
    """

    duration: float

    @abstractmethod
    def targets(self) -> list[Blade]:
        """Return all blades this action affects."""
        pass


@dataclass
class DrawAction(Action):
    """
    Draw a blade (fade it into visibility).

    The blade's opacity increases from 0 to 1 over the duration.
    """

    blade: Blade
    mode: DrawMode = "fade_in"
    smoother: Smoother = field(default_factory=lambda: smooth_linear)
    color: Color | None = None  # Optional color override

    def targets(self) -> list[Blade]:
        return [self.blade]

    def opacity_at(self, local_time: float) -> float:
        """Get opacity at time t within this action."""
        if local_time <= 0:
            return 0.0
        if local_time >= self.duration:
            return 1.0

        # For fade_in, opacity increases linearly (or with smoothing)
        # Smoothing gives velocity, integrate to get position
        progress = local_time / self.duration
        return progress

    def scale_at(self, local_time: float) -> float:
        """Get scale at time t (for scale_in mode)."""
        if self.mode != "scale_in":
            return 1.0

        if local_time <= 0:
            return 0.0
        if local_time >= self.duration:
            return 1.0

        return local_time / self.duration


@dataclass
class HideAction(Action):
    """
    Hide a blade (fade it out of visibility).

    The blade's opacity decreases from 1 to 0 over the duration.
    """

    blade: Blade
    mode: HideMode = "fade_out"
    smoother: Smoother = field(default_factory=lambda: smooth_linear)

    def targets(self) -> list[Blade]:
        return [self.blade]

    def opacity_at(self, local_time: float) -> float:
        """Get opacity at time t within this action."""
        if local_time <= 0:
            return 1.0
        if local_time >= self.duration:
            return 0.0

        progress = local_time / self.duration
        return 1.0 - progress


@dataclass
class ReplaceAction(Action):
    """
    Atomically replace one blade with another.

    The old blade fades out while the new blade fades in simultaneously.
    """

    old_blade: Blade
    new_blade: Blade
    smoother: Smoother = field(default_factory=lambda: smooth_linear)
    new_color: Color | None = None  # Optional color for the new blade

    def targets(self) -> list[Blade]:
        return [self.old_blade, self.new_blade]

    def old_opacity_at(self, local_time: float) -> float:
        """Get old blade opacity."""
        if local_time <= 0:
            return 1.0
        if local_time >= self.duration:
            return 0.0
        return 1.0 - local_time / self.duration

    def new_opacity_at(self, local_time: float) -> float:
        """Get new blade opacity."""
        if local_time <= 0:
            return 0.0
        if local_time >= self.duration:
            return 1.0
        return local_time / self.duration


@dataclass
class RotateAction(Action):
    """
    Rotate a blade in a given plane (bivector).

    Uses Motor.rotor() with the bivector defining the rotation plane.
    The rotation is distributed smoothly over the duration.

    In proper GA, rotations are defined by bivectors (oriented planes),
    not by axis vectors.
    """

    blade: Blade
    plane: Blade  # Bivector defining rotation plane
    angle: float
    center: NDArray | None = None
    smoother: Smoother = field(default_factory=lambda: smooth_linear)

    def targets(self) -> list[Blade]:
        return [self.blade]

    def angle_at(self, local_time: float) -> float:
        """Get cumulative rotation angle at time t, with smoothing applied."""
        if local_time <= 0:
            return 0.0
        if local_time >= self.duration:
            return self.angle

        # Integrate the smoother from 0 to local_time to get cumulative progress
        # For now, use simple trapezoidal approximation
        n_steps = 100
        dt = local_time / n_steps
        cumulative = 0.0
        for i in range(n_steps):
            t = i * dt
            cumulative += self.smoother(t, self.duration) * dt

        return self.angle * cumulative


@dataclass
class ScrewAction(Action):
    """
    Apply a screw motion: rotate in plane + translate.

    Uses Motor.screw() with bivector for rotation plane and explicit
    translation vector.

    In proper GA, the rotation plane (bivector) is the fundamental object.
    """

    blade: Blade
    plane: Blade  # Bivector defining rotation plane
    angle: float
    translation: NDArray  # Explicit translation vector
    center: NDArray | None = None
    smoother: Smoother = field(default_factory=lambda: smooth_linear)

    def targets(self) -> list[Blade]:
        return [self.blade]

    def progress_at(self, local_time: float) -> float:
        """Get progress [0, 1] at time t, with smoothing applied."""
        if local_time <= 0:
            return 0.0
        if local_time >= self.duration:
            return 1.0

        # Integrate the smoother from 0 to local_time
        n_steps = 100
        dt = local_time / n_steps
        cumulative = 0.0
        for i in range(n_steps):
            t = i * dt
            cumulative += self.smoother(t, self.duration) * dt

        return cumulative


@dataclass
class RotateToPlaneAction(Action):
    """
    Rotate a blade into a target plane.

    Computes the rotation needed to align the blade's normal with the
    target plane normal, then applies it smoothly.
    """

    blade: Blade
    target_normal: NDArray
    smoother: Smoother = field(default_factory=lambda: smooth_linear)

    def targets(self) -> list[Blade]:
        return [self.blade]

    def progress_at(self, local_time: float) -> float:
        """Get progress [0, 1] at time t, with smoothing applied."""
        if local_time <= 0:
            return 0.0
        if local_time >= self.duration:
            return 1.0

        # Integrate the smoother from 0 to local_time
        n_steps = 100
        dt = local_time / n_steps
        cumulative = 0.0
        for i in range(n_steps):
            t = i * dt
            cumulative += self.smoother(t, self.duration) * dt

        return cumulative


@dataclass
class WaitAction(Action):
    """
    Hold the current state for a duration (no changes).
    """

    def targets(self) -> list[Blade]:
        return []


class Timeline:
    """
    A sequence of animation actions.

    Build animations declaratively:
        timeline = Timeline()
        timeline.draw(e1, duration=1.0)
        timeline.rotate(e123, axis=[1,1,1], angle=2*pi, duration=4.0)

    The timeline can be played by AnimatedCanvas:
        canvas.play(timeline)

    Method chaining is supported:
        timeline.draw(e1).draw(e2).rotate(e123, ...)
    """

    def __init__(self):
        self._actions: list[Action] = []

    @property
    def actions(self) -> list[Action]:
        """List of all actions in the timeline."""
        return self._actions

    @property
    def total_duration(self) -> float:
        """Total duration of the timeline."""
        return sum(a.duration for a in self._actions)

    def draw(
        self,
        blade: Blade,
        duration: float = 1.0,
        mode: DrawMode = "fade_in",
        smooth: str | Smoother = "linear",
        color: Color | None = None,
    ) -> "Timeline":
        """
        Add a draw action (blade fades into visibility).

        Args:
            blade: The blade to draw
            duration: Time to complete the draw
            mode: "fade_in" or "scale_in"
            smooth: Smoother name or function
            color: Optional color override (RGB tuple, 0-1 range)

        Returns:
            Self for chaining
        """
        smoother = get_smoother(smooth) if isinstance(smooth, str) else smooth
        self._actions.append(
            DrawAction(
                duration=duration,
                blade=blade,
                mode=mode,
                smoother=smoother,
                color=color,
            )
        )
        return self

    def hide(
        self,
        blade: Blade,
        duration: float = 1.0,
        mode: HideMode = "fade_out",
        smooth: str | Smoother = "linear",
    ) -> "Timeline":
        """
        Add a hide action (blade fades out of visibility).

        Args:
            blade: The blade to hide
            duration: Time to complete the hide
            mode: "fade_out" or "scale_out"
            smooth: Smoother name or function

        Returns:
            Self for chaining
        """
        smoother = get_smoother(smooth) if isinstance(smooth, str) else smooth
        self._actions.append(
            HideAction(
                duration=duration,
                blade=blade,
                mode=mode,
                smoother=smoother,
            )
        )
        return self

    def replace(
        self,
        old_blade: Blade,
        new_blade: Blade,
        duration: float = 1.0,
        smooth: str | Smoother = "linear",
        new_color: Color | None = None,
    ) -> "Timeline":
        """
        Add a replace action (old blade out, new blade in).

        Args:
            old_blade: The blade to hide
            new_blade: The blade to show
            duration: Time to complete the replacement
            smooth: Smoother name or function
            new_color: Optional color for the new blade

        Returns:
            Self for chaining
        """
        smoother = get_smoother(smooth) if isinstance(smooth, str) else smooth
        self._actions.append(
            ReplaceAction(
                duration=duration,
                old_blade=old_blade,
                new_blade=new_blade,
                smoother=smoother,
                new_color=new_color,
            )
        )
        return self

    def rotate(
        self,
        blade: Blade,
        plane: Blade,
        angle: float,
        duration: float = 1.0,
        center: tuple | list | NDArray | None = None,
        smooth: str | Smoother = "linear",
    ) -> "Timeline":
        """
        Add a rotation action in a given plane.

        In proper GA, rotations are defined by bivectors (oriented planes),
        not by axis vectors. The bivector B defines the plane of rotation.

        Args:
            blade: The blade to rotate
            plane: Bivector defining the rotation plane (e.g., e1 ^ e2)
            angle: Total rotation angle in radians
            duration: Time to complete the rotation
            center: Optional center of rotation (default: origin)
            smooth: Smoother name or function

        Returns:
            Self for chaining

        Example:
            e1, e2, e3 = basis_vectors(dim=3)
            timeline.rotate(q, plane=e1^e2, angle=pi/2, duration=2.0)
        """
        if plane.grade != 2:
            raise ValueError(f"plane must be a bivector (grade 2), got grade {plane.grade}")

        smoother = get_smoother(smooth) if isinstance(smooth, str) else smooth
        self._actions.append(
            RotateAction(
                duration=duration,
                blade=blade,
                plane=plane,
                angle=angle,
                center=array(center, dtype=float) if center is not None else None,
                smoother=smoother,
            )
        )
        return self

    def screw(
        self,
        blade: Blade,
        plane: Blade,
        angle: float,
        translation: tuple | list | NDArray,
        duration: float = 1.0,
        center: tuple | list | NDArray | None = None,
        smooth: str | Smoother = "linear",
    ) -> "Timeline":
        """
        Add a screw motion action (rotate in plane + translate).

        In proper GA, the rotation plane (bivector) is the fundamental object.
        The translation is specified explicitly as a vector.

        Args:
            blade: The blade to transform
            plane: Bivector defining the rotation plane
            angle: Total rotation angle in radians
            translation: Translation vector
            duration: Time to complete the screw motion
            center: Optional center point
            smooth: Smoother name or function

        Returns:
            Self for chaining

        Example:
            e1, e2, e3 = basis_vectors(dim=3)
            # Rotate in xy-plane while translating along z
            timeline.screw(q, plane=e1^e2, angle=2*pi,
                          translation=[0, 0, 1], duration=4.0)
        """
        if plane.grade != 2:
            raise ValueError(f"plane must be a bivector (grade 2), got grade {plane.grade}")

        smoother = get_smoother(smooth) if isinstance(smooth, str) else smooth
        self._actions.append(
            ScrewAction(
                duration=duration,
                blade=blade,
                plane=plane,
                angle=angle,
                translation=array(translation, dtype=float),
                center=array(center, dtype=float) if center is not None else None,
                smoother=smoother,
            )
        )
        return self

    def rotate_to_plane(
        self,
        blade: Blade,
        normal: tuple | list | NDArray,
        duration: float = 1.0,
        smooth: str | Smoother = "linear",
    ) -> "Timeline":
        """
        Add a rotation that aligns blade with a target plane.

        The blade is rotated so its orientation aligns with the plane
        defined by the given normal vector.

        Args:
            blade: The blade to rotate
            normal: Normal vector of target plane
            duration: Time to complete the rotation
            smooth: Smoother name or function

        Returns:
            Self for chaining
        """
        smoother = get_smoother(smooth) if isinstance(smooth, str) else smooth
        self._actions.append(
            RotateToPlaneAction(
                duration=duration,
                blade=blade,
                target_normal=array(normal, dtype=float),
                smoother=smoother,
            )
        )
        return self

    def wait(self, duration: float = 1.0) -> "Timeline":
        """
        Add a pause (hold current state).

        Args:
            duration: Time to wait

        Returns:
            Self for chaining
        """
        self._actions.append(WaitAction(duration=duration))
        return self

    def action_at(self, time: float) -> tuple[Action | None, float]:
        """
        Get the action active at a given time.

        Args:
            time: Time from start of timeline

        Returns:
            (action, local_time) tuple where local_time is time within that action.
            Returns (None, 0) if time is past the end.
        """
        if time < 0:
            return (self._actions[0] if self._actions else None, 0.0)

        elapsed = 0.0
        for action in self._actions:
            if time < elapsed + action.duration:
                return (action, time - elapsed)
            elapsed += action.duration

        # Past end
        return (None, 0.0)

    def all_blades(self) -> list[Blade]:
        """Get all unique blades referenced in the timeline."""
        seen: dict[int, Blade] = {}
        for action in self._actions:
            for blade in action.targets():
                blade_id = id(blade)
                if blade_id not in seen:
                    seen[blade_id] = blade
        return list(seen.values())
