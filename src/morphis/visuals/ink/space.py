"""
Organic Space

The abstract enclosing space of a conceptual figure: a phase space, a Hilbert
space, a configuration space, drawn as a lumpy closed surface we can see into.
The shape is a star-shaped surface whose radius varies smoothly with direction.
Every random choice flows from one seed, so a figure re-renders identically,
and a different seed or parameters gives a different but equally organic shape.
"""

from __future__ import annotations

from numpy import (
    arange,
    arctan2,
    asarray,
    bincount,
    concatenate,
    convolve,
    cos,
    einsum,
    linspace,
    maximum as elementwise_max,
    meshgrid,
    ones,
    pi,
    sin,
    sqrt,
    stack,
)
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.visuals.ink.camera import Camera


class OrganicSpace(BaseModel):
    """
    A seeded, lumpy, star-shaped surface.

    The radius along a unit direction u is

        r(u) = 1 + lumpiness * sum_k a_k cos(π f_k (u · d_k) + φ_k) / sum_k a_k

    with random directions d_k, frequencies f_k in [1, detail], phases φ_k, and
    amplitudes a_k falling as 1 / f_k so the large bulges dominate.

    Attributes:
        seed: Fixes the shape; the same seed always gives the same space
        lumpiness: Size of the bulges relative to the mean radius
        detail: Highest bulge frequency; larger gives more, smaller lobes
        lobes: Number of superposed bulge terms
        stretch: Axis scales applied after shaping, for an elongated space
        size: Mean radius in drawing units
        center: Center in drawing coordinates
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    lumpiness: float = 0.28
    detail: float = 3.0
    lobes: int = 9
    stretch: tuple[float, float, float] = (1.5, 1.15, 1.0)
    size: float = 3.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _terms(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Draw the bulge directions, frequencies, phases, and amplitudes."""
        rng = default_rng(self.seed)
        directions = rng.normal(size=(self.lobes, 3))
        directions = directions / norm(directions, axis=-1, keepdims=True)
        frequencies = rng.uniform(1.0, max(self.detail, 1.0), size=self.lobes)
        phases = rng.uniform(0.0, 2 * pi, size=self.lobes)
        amplitudes = 1.0 / frequencies

        return directions, frequencies, phases, amplitudes

    def radius(self, directions: NDArray) -> NDArray:
        """Radial scale factor r(u) for unit directions of shape (..., 3)."""
        d, f, phase, a = self._terms()
        alignment = einsum("...c, kc -> ...k", directions, d)
        bulge = einsum("...k, k -> ...", cos(pi * f * alignment + phase), a) / a.sum()
        result = 1.0 + self.lumpiness * bulge

        return result

    def surface(self, n_polar: int = 96, n_azimuthal: int = 192) -> NDArray:
        """Sample the surface on a polar grid; returns shape (n_polar, n_azimuthal, 3)."""
        polar, azimuthal = meshgrid(linspace(0, pi, n_polar), linspace(0, 2 * pi, n_azimuthal), indexing="ij")
        directions = stack([sin(polar) * cos(azimuthal), sin(polar) * sin(azimuthal), cos(polar)], axis=-1)
        points = self.place(directions)

        return points

    def place(self, directions: NDArray) -> NDArray:
        """Surface points along unit directions of shape (..., 3)."""
        scale = self.size * self.radius(directions)[..., None] * asarray(self.stretch)
        points = asarray(self.center) + scale * directions

        return points

    def samples(self, count: int = 40000) -> NDArray:
        """Near-uniform surface samples along a Fibonacci spiral of directions; shape (count, 3)."""
        n = arange(count) + 0.5
        height = 1.0 - 2.0 * n / count
        ring = sqrt(1.0 - height**2)
        turn = pi * (3.0 - sqrt(5.0)) * n
        directions = stack([ring * cos(turn), ring * sin(turn), height], axis=-1)
        points = self.place(directions)

        return points

    def outline(self, camera: Camera, n_angles: int = 720, smoothing: int = 15) -> tuple[NDArray, NDArray]:
        """
        Silhouette of the space as seen by the camera.

        The projected surface is binned by angle about the projected center and
        the farthest point in each bin kept, which traces the silhouette of any
        star-shaped blob. A short circular moving average removes binning jitter.

        Returns:
            outline: Closed polyline of shape (n_angles + 1, 2) on the page
            center: Projected center of the space, shape (2,)
        """
        page, _ = camera.project(self.samples())
        center, _ = camera.project(asarray(self.center))
        offset = page - center
        angle = arctan2(offset[:, 1], offset[:, 0])
        bins = ((angle + pi) / (2 * pi) * n_angles).astype(int) % n_angles
        reach = bincount(bins, minlength=n_angles) * 0.0
        elementwise_max.at(reach, bins, norm(offset, axis=-1))

        half = smoothing // 2
        padded = concatenate([reach[-half:], reach, reach[:half]])
        smooth = convolve(padded, ones(2 * half + 1) / (2 * half + 1), mode="valid")

        theta = -pi + (linspace(0, n_angles - 1, n_angles) + 0.5) * 2 * pi / n_angles
        ring = center + smooth[:, None] * stack([cos(theta), sin(theta)], axis=-1)
        closed = concatenate([ring, ring[:1]])

        return closed, center
