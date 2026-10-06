"""
Sketch Camera

Maps 3D drawing coordinates to the 2D page. The camera orbits a focal point,
with z as the world up direction. A field of view of zero gives an orthographic
projection; a small field of view gives the gentle perspective of a hand
drawing.
"""

from __future__ import annotations

from numpy import arcsin, arctan2, array, asarray, clip, cos, cross, deg2rad, einsum, rad2deg, sin, stack, tan
from numpy.linalg import norm
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class Camera(BaseModel):
    """
    An orbiting camera.

    Attributes:
        azimuth: Rotation about world z, in degrees (0 looks along +y)
        elevation: Angle above the xy-plane, in degrees
        distance: Distance from the focal point
        focal_point: Point the camera looks at
        fov: Vertical field of view in degrees; 0 for orthographic
    """

    model_config = ConfigDict(frozen=True)

    azimuth: float = -60.0
    elevation: float = 20.0
    distance: float = 12.0
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 18.0

    @classmethod
    def looking(
        cls,
        position: tuple[float, float, float],
        focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 18.0,
    ) -> Camera:
        """Build a camera from an eye position and focal point."""
        offset = asarray(position, dtype=float) - asarray(focal_point, dtype=float)
        distance = float(norm(offset))
        elevation = float(rad2deg(arcsin(clip(offset[2] / distance, -1.0, 1.0))))
        azimuth = float(rad2deg(arctan2(offset[0], -offset[1])))
        camera = cls(azimuth=azimuth, elevation=elevation, distance=distance, focal_point=focal_point, fov=fov)

        return camera

    @property
    def view_direction(self) -> NDArray:
        """Unit vector from the focal point toward the eye."""
        a = deg2rad(self.azimuth)
        e = deg2rad(self.elevation)
        direction = array([cos(e) * sin(a), -cos(e) * cos(a), sin(e)])

        return direction

    @property
    def frame(self) -> NDArray:
        """Rows are the page right, page up, and toward-eye unit vectors."""
        toward = self.view_direction
        right = cross(array([0.0, 0.0, 1.0]), toward)
        right = right / norm(right)
        up = cross(toward, right)
        basis = stack([right, up, toward])

        return basis

    def project(self, points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Project 3D points to the page.

        Args:
            points: Array of shape (..., 3)

        Returns:
            page: Array of shape (..., 2), in units of the focal plane
            depth: Array of shape (...,), larger is farther from the eye
        """
        relative = asarray(points, dtype=float) - asarray(self.focal_point)
        local = einsum("ab, ...b -> ...a", self.frame, relative)
        depth = self.distance - local[..., 2]
        scale = self.distance / depth if self.fov > 0 else 1.0 + 0.0 * depth
        page = local[..., :2] * scale[..., None]

        return page, depth

    @property
    def half_height(self) -> float:
        """Half-height of the focal plane visible through the field of view."""
        half = self.distance * tan(deg2rad(0.5 * self.fov)) if self.fov > 0 else 0.5 * self.distance

        return float(half)
