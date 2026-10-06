"""
Blade Depictions

Stateless pictures of bare bivectors and trivectors, computed from the element
alone. A bivector has a plane, a magnitude, and an orientation, but no shape, so
it is drawn as an oriented disk: a disk in its plane whose area is |B|, with a
circulation arrow showing the sense. A trivector in three dimensions has a
magnitude and a sign, so it is drawn as a ball whose volume is |T|, with a
circulation arrow around its equator showing the handedness.

Both pictures are symmetric about their axis, so a smoothly changing element
gives a smoothly changing picture. The one choice that needs a direction, where
on the rim the circulation arrow sits, uses a fixed reference direction rather
than any factorization of the element, so it cannot jump as the element turns.

All geometry is plain arrays in drawing coordinates; nothing here touches a
renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy import array, cbrt, concatenate, cos, cross, linspace, meshgrid, pi, sin, sqrt, stack, zeros
from numpy.linalg import norm
from numpy.typing import NDArray


# Where the circulation arrow sits: the rim point farthest along this direction.
# When a disk faces straight along it, the fallback direction is used instead.
REFERENCE_DIRECTION = array([0.0, 0.0, 1.0])
FALLBACK_DIRECTION = array([1.0, 0.0, 0.0])

# Size of the circulation arrowhead as fractions of the radius
HEAD_LENGTH = 0.28
HEAD_RADIUS = 0.09
HEAD_SEGMENTS = 16


@dataclass
class OrientedShape:
    """
    Geometry of an oriented disk or ball.

    Attributes:
        surface: Vertices of the filled surface, shape (n, 3)
        faces: Face connectivity in indexed format; fixed for a given resolution
        rim: Closed polyline around the rim or equator, shape (m, 3)
        head: Vertices of the circulation arrowhead, a cone on the rim pointing in the sense of the orientation
        head_faces: Face connectivity of the arrowhead; fixed
    """

    surface: NDArray
    faces: NDArray
    rim: NDArray
    head: NDArray
    head_faces: NDArray


def bivector_normal(B: NDArray) -> NDArray:
    """
    Normal of a 3D bivector, with |normal| = |B|.

    For B = Σ_{a<b} B^{ab} e_ab the normal is (B^{23}, B^{31}, B^{12}), so that
    e_1 ∧ e_2 has normal e_3 and the circulation from e_1 toward e_2 is
    counterclockwise about it.
    """
    normal = array([B[1, 2], B[2, 0], B[0, 1]], dtype=float)

    return normal


def _in_plane_frame(normal: NDArray) -> tuple[NDArray, NDArray]:
    """
    Orthonormal pair (d, w) spanning the plane normal to `normal`, with w = n × d.

    d is the reference direction projected into the plane, so it depends only
    on the plane, continuously, except where the plane faces the reference
    direction head-on.
    """
    reference = REFERENCE_DIRECTION - (REFERENCE_DIRECTION @ normal) * normal
    is_degenerate = norm(reference) < 1e-6
    reference = FALLBACK_DIRECTION - (FALLBACK_DIRECTION @ normal) * normal if is_degenerate else reference
    d = reference / norm(reference)
    w = cross(normal, d)

    return d, w


def _ring(center: NDArray, radius: float, d: NDArray, w: NDArray, segments: int) -> NDArray:
    """Closed circle of the given radius in the plane of d and w, starting at d."""
    angles = linspace(0.0, 2 * pi, segments + 1)
    points = center + radius * (cos(angles)[:, None] * d + sin(angles)[:, None] * w)

    return points


def oriented_disk(B: NDArray, center: NDArray, segments: int = 64) -> OrientedShape:
    """
    Oriented disk of a 3D bivector: area |B| in its plane, circulating with its sense.

    Args:
        B: Bivector components in 3D, antisymmetric array of shape (3, 3)
        center: Disk center in drawing coordinates, shape (3,)
        segments: Rim resolution

    Returns:
        The disk geometry; a zero bivector gives a disk of radius zero
    """
    center = array(center, dtype=float)
    normal = bivector_normal(B)
    magnitude = float(norm(normal))
    unit = normal / magnitude if magnitude > 1e-12 else REFERENCE_DIRECTION
    radius = sqrt(magnitude / pi)
    d, w = _in_plane_frame(unit)

    rim = _ring(center, radius, d, w, segments)
    surface = concatenate([center[None, :], rim[:-1]])
    fan = [[3, 0, 1 + k, 1 + (k + 1) % segments] for k in range(segments)]
    faces = array(fan).ravel()

    # The arrowhead sits at the rim point along d and points along w, the sense of the orientation
    head, head_faces = _arrowhead(center + radius * d, w, unit, radius)

    return OrientedShape(surface, faces, rim, head, head_faces)


def oriented_ball(T: float, center: NDArray, segments: int = 32) -> OrientedShape:
    """
    Oriented ball of a 3D trivector T e_123: volume |T|, with handedness shown by its equator.

    The equator lies in the drawing's horizontal plane. The circulation arrow
    runs counterclockwise about +z when T > 0 (right-handed, like e_123) and
    clockwise when T < 0.

    Args:
        T: Trivector coefficient along e_123
        center: Ball center in drawing coordinates, shape (3,)
        segments: Longitude resolution; latitude uses half as many

    Returns:
        The ball geometry; a zero trivector gives a ball of radius zero
    """
    center = array(center, dtype=float)
    radius = float(cbrt(3 * abs(T) / (4 * pi)))
    sense = 1.0 if T >= 0 else -1.0

    rings = segments // 2
    polar, azimuthal = meshgrid(
        linspace(0.0, pi, rings + 1), linspace(0.0, 2 * pi, segments, endpoint=False), indexing="ij"
    )
    directions = stack([sin(polar) * cos(azimuthal), sin(polar) * sin(azimuthal), cos(polar)], axis=-1)
    surface = center + radius * directions.reshape(-1, 3)

    quads = []
    for m in range(rings):
        for n in range(segments):
            a = m * segments + n
            b = m * segments + (n + 1) % segments
            quads.append([4, a, b, b + segments, a + segments])
    faces = array(quads).ravel()

    d = array([1.0, 0.0, 0.0])
    w = sense * array([0.0, 1.0, 0.0])
    rim = _ring(center, radius, d, w, segments * 2)
    head, head_faces = _arrowhead(center + radius * d, w, array([0.0, 0.0, 1.0]), radius)

    return OrientedShape(surface, faces, rim, head, head_faces)


def _arrowhead(point: NDArray, direction: NDArray, normal: NDArray, radius: float) -> tuple[NDArray, NDArray]:
    """
    Cone centered on a rim point, pointing along the unit tangent `direction`.

    Its size scales with the radius of the disk or ball it marks. The vertex
    count and faces are fixed, so the cone can be moved in place frame to frame.
    """
    length = HEAD_LENGTH * radius
    width = HEAD_RADIUS * radius
    side = cross(direction, normal)
    tip = point + 0.5 * length * direction
    base = point - 0.5 * length * direction

    angles = linspace(0.0, 2 * pi, HEAD_SEGMENTS, endpoint=False)
    ring = base + width * (cos(angles)[:, None] * normal + sin(angles)[:, None] * side)
    vertices = concatenate([tip[None, :], base[None, :], ring])

    sides = [[3, 0, 2 + k, 2 + (k + 1) % HEAD_SEGMENTS] for k in range(HEAD_SEGMENTS)]
    cap = [[3, 1, 2 + (k + 1) % HEAD_SEGMENTS, 2 + k] for k in range(HEAD_SEGMENTS)]
    faces = array(sides + cap).ravel()

    return vertices, faces


def pad_to_3d(components: NDArray, grade: int) -> NDArray:
    """Embed the components of a grade-1, 2, or 3 element of dimension at most 3 into 3D."""
    dim = components.shape[-1]
    padded = zeros((3,) * grade) if grade > 0 else zeros(())
    index = tuple(slice(0, dim) for _ in range(grade))
    padded[index] = components

    return padded
