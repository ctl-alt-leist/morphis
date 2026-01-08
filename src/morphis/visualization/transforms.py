"""
Blade Transforms for Animation

Provides smooth transformations (rotation, translation, scaling) for visualizing
blades in motion. Supports sequenced animations with easing functions.

Key classes:
- BladeTransform: Combines rotation, translation, and scale
- AnimationSequence: Chain multiple transforms over time
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import vtk
from numpy import array, cos, eye, ndarray, pi, zeros
from numpy.linalg import norm

from morphis.core.rotations import rotation_matrix


# =============================================================================
# Easing Functions
# =============================================================================


def ease_linear(t: float) -> float:
    """Linear interpolation (no easing)."""
    return t


def ease_in_out_cubic(t: float) -> float:
    """Smooth start and end (cubic)."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - (-2 * t + 2) ** 3 / 2


def ease_in_out_sine(t: float) -> float:
    """Smooth sinusoidal easing."""
    return -(cos(pi * t) - 1) / 2


def ease_out_quad(t: float) -> float:
    """Decelerate at end."""
    return 1 - (1 - t) ** 2


def ease_in_quad(t: float) -> float:
    """Accelerate from start."""
    return t * t


# =============================================================================
# Transform State
# =============================================================================


@dataclass
class BladeTransform:
    """
    Represents a complete transform state for a blade visualization.

    Combines rotation (about an axis through a point) and translation.
    """

    # Rotation
    angle: float = 0.0  # radians
    axis: ndarray = field(default_factory=lambda: array([0.0, 0.0, 1.0]))
    rotation_center: ndarray = field(default_factory=lambda: zeros(3))

    # Translation
    translation: ndarray = field(default_factory=lambda: zeros(3))

    # Scale (uniform)
    scale: float = 1.0

    def to_vtk_transform(self) -> vtk.vtkTransform:
        """Convert to VTK transform for actor manipulation."""
        transform = vtk.vtkTransform()
        transform.PostMultiply()

        # Order: translate to origin, scale, rotate, translate back, then final translate
        # Move rotation center to origin
        transform.Translate(-self.rotation_center[0], -self.rotation_center[1], -self.rotation_center[2])

        # Scale
        if self.scale != 1.0:
            transform.Scale(self.scale, self.scale, self.scale)

        # Rotate (VTK uses degrees)
        angle_deg = self.angle * 180.0 / pi
        transform.RotateWXYZ(angle_deg, self.axis[0], self.axis[1], self.axis[2])

        # Move back from origin
        transform.Translate(self.rotation_center[0], self.rotation_center[1], self.rotation_center[2])

        # Apply final translation
        transform.Translate(self.translation[0], self.translation[1], self.translation[2])

        return transform

    def to_matrix(self) -> ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        # Build rotation matrix
        R = rotation_matrix(self.angle, self.axis)

        # Full 4x4 matrix
        M = eye(4)
        M[:3, :3] = self.scale * R
        M[:3, 3] = self.translation + self.rotation_center - R @ self.rotation_center

        return M

    @staticmethod
    def interpolate(t1: "BladeTransform", t2: "BladeTransform", alpha: float) -> "BladeTransform":
        """
        Linearly interpolate between two transforms.

        Args:
            t1: Start transform
            t2: End transform
            alpha: Interpolation factor (0 = t1, 1 = t2)

        Returns:
            Interpolated transform
        """
        return BladeTransform(
            angle=t1.angle + alpha * (t2.angle - t1.angle),
            axis=t1.axis + alpha * (t2.axis - t1.axis),  # Works if axes are same
            rotation_center=t1.rotation_center + alpha * (t2.rotation_center - t1.rotation_center),
            translation=t1.translation + alpha * (t2.translation - t1.translation),
            scale=t1.scale + alpha * (t2.scale - t1.scale),
        )


# =============================================================================
# Animation Segments
# =============================================================================


@dataclass
class AnimationSegment:
    """
    A single segment of an animation sequence.

    Interpolates from start_transform to end_transform over duration seconds.
    """

    start_transform: BladeTransform
    end_transform: BladeTransform
    duration: float  # seconds
    easing: Callable[[float], float] = ease_linear

    def evaluate(self, local_time: float) -> BladeTransform:
        """
        Evaluate transform at local time within this segment.

        Args:
            local_time: Time in seconds from start of this segment

        Returns:
            Interpolated transform
        """
        if local_time <= 0:
            return self.start_transform
        if local_time >= self.duration:
            return self.end_transform

        # Normalized time [0, 1]
        t = local_time / self.duration
        alpha = self.easing(t)

        return BladeTransform.interpolate(self.start_transform, self.end_transform, alpha)


class AnimationSequence:
    """
    A sequence of animation segments played in order.

    Example:
        seq = AnimationSequence()
        seq.rotate(axis=[1,1,1], angle=4*pi, duration=8.0)
        seq.translate(target=[1,1,1], duration=4.0)
        seq.rotate(axis=[1,1,1], angle=2*pi, duration=4.0)

        # In animation loop:
        transform = seq.evaluate(current_time)
    """

    def __init__(self):
        self.segments: List[AnimationSegment] = []
        self._current_state = BladeTransform()

    @property
    def total_duration(self) -> float:
        """Total duration of all segments."""
        return sum(seg.duration for seg in self.segments)

    def rotate(
        self,
        angle: float,
        axis: Tuple[float, float, float] = (0, 0, 1),
        duration: float = 1.0,
        center: Tuple[float, float, float] = (0, 0, 0),
        easing: Callable[[float], float] = ease_linear,
    ) -> "AnimationSequence":
        """
        Add a rotation segment.

        Args:
            angle: Rotation angle in radians
            axis: Rotation axis (will be normalized)
            duration: Duration in seconds
            center: Center of rotation
            easing: Easing function

        Returns:
            Self for chaining
        """
        axis_arr = array(axis, dtype=float)
        axis_arr = axis_arr / norm(axis_arr)
        center_arr = array(center, dtype=float)

        start = BladeTransform(
            angle=self._current_state.angle,
            axis=axis_arr,
            rotation_center=center_arr,
            translation=self._current_state.translation.copy(),
            scale=self._current_state.scale,
        )

        end = BladeTransform(
            angle=self._current_state.angle + angle,
            axis=axis_arr,
            rotation_center=center_arr,
            translation=self._current_state.translation.copy(),
            scale=self._current_state.scale,
        )

        self.segments.append(AnimationSegment(start, end, duration, easing))
        self._current_state = end

        return self

    def translate(
        self,
        target: Tuple[float, float, float],
        duration: float = 1.0,
        easing: Callable[[float], float] = ease_in_out_cubic,
    ) -> "AnimationSequence":
        """
        Add a translation segment.

        Args:
            target: Target position (absolute, not relative)
            duration: Duration in seconds
            easing: Easing function

        Returns:
            Self for chaining
        """
        target_arr = array(target, dtype=float)

        start = BladeTransform(
            angle=self._current_state.angle,
            axis=self._current_state.axis.copy(),
            rotation_center=self._current_state.rotation_center.copy(),
            translation=self._current_state.translation.copy(),
            scale=self._current_state.scale,
        )

        end = BladeTransform(
            angle=self._current_state.angle,
            axis=self._current_state.axis.copy(),
            rotation_center=self._current_state.rotation_center.copy(),
            translation=target_arr,
            scale=self._current_state.scale,
        )

        self.segments.append(AnimationSegment(start, end, duration, easing))
        self._current_state = end

        return self

    def scale_to(
        self,
        scale: float,
        duration: float = 1.0,
        easing: Callable[[float], float] = ease_in_out_cubic,
    ) -> "AnimationSequence":
        """
        Add a scaling segment.

        Args:
            scale: Target scale factor
            duration: Duration in seconds
            easing: Easing function

        Returns:
            Self for chaining
        """
        start = BladeTransform(
            angle=self._current_state.angle,
            axis=self._current_state.axis.copy(),
            rotation_center=self._current_state.rotation_center.copy(),
            translation=self._current_state.translation.copy(),
            scale=self._current_state.scale,
        )

        end = BladeTransform(
            angle=self._current_state.angle,
            axis=self._current_state.axis.copy(),
            rotation_center=self._current_state.rotation_center.copy(),
            translation=self._current_state.translation.copy(),
            scale=scale,
        )

        self.segments.append(AnimationSegment(start, end, duration, easing))
        self._current_state = end

        return self

    def wait(self, duration: float) -> "AnimationSequence":
        """
        Add a pause (no change) segment.

        Args:
            duration: Duration in seconds

        Returns:
            Self for chaining
        """
        start = BladeTransform(
            angle=self._current_state.angle,
            axis=self._current_state.axis.copy(),
            rotation_center=self._current_state.rotation_center.copy(),
            translation=self._current_state.translation.copy(),
            scale=self._current_state.scale,
        )

        self.segments.append(AnimationSegment(start, start, duration, ease_linear))

        return self

    def evaluate(self, time: float) -> BladeTransform:
        """
        Evaluate the animation at a given time.

        Args:
            time: Time in seconds from start of animation

        Returns:
            Transform at that time
        """
        if not self.segments:
            return BladeTransform()

        if time <= 0:
            return self.segments[0].start_transform

        # Find which segment we're in
        elapsed = 0.0
        for segment in self.segments:
            if time < elapsed + segment.duration:
                local_time = time - elapsed
                return segment.evaluate(local_time)
            elapsed += segment.duration

        # Past end - return final state
        return self.segments[-1].end_transform

    def get_vtk_transform(self, time: float) -> vtk.vtkTransform:
        """Get VTK transform for a given time."""
        return self.evaluate(time).to_vtk_transform()
