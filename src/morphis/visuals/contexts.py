"""
Context-Specific Vector Visualization

Rendering functions for blades in specific geometric algebra contexts:
- PGA (Projective Geometric Algebra): Points, lines, planes as geometric entities
- CGA (Conformal Geometric Algebra): Spheres, circles, etc. (future)

In PGA, blades represent geometric objects that extend beyond their local
representation. Points are at specific locations, lines extend infinitely,
planes are infinite surfaces.
"""

from numpy import array, cross, zeros
from numpy.linalg import norm as np_norm
from numpy.typing import NDArray

from morphis.elements.metric import GAStructure
from morphis.elements.vector import Vector
from morphis.transforms.projective import bulk, euclidean, is_point
from morphis.visuals.canvas import Canvas
from morphis.visuals.drawing.vectors import VectorStyle


# =============================================================================
# PGA Style Configuration
# =============================================================================


class PGAStyle(VectorStyle):
    """Style parameters for PGA visualization."""

    # Point rendering
    point_radius: float = 0.04
    direction_arrow_length: float = 0.5

    # Line rendering
    line_extent: float = 2.0
    line_radius: float = 0.01

    # Plane rendering
    plane_extent: float = 2.0


# =============================================================================
# PGA Point Rendering
# =============================================================================


def render_pga_point(
    blade: Vector,
    canvas: Canvas,
    style: PGAStyle | None = None,
) -> None:
    """
    Render PGA point (grade-1 in PGA) as sphere at Euclidean location.

    Points have unit weight (e_0 = 1) and render as spheres.
    Directions have zero weight (e_0 = 0) and render as arrows from origin.
    """
    if blade.grade != 1:
        raise ValueError(f"render_pga_point requires grade-1, got {blade.grade}")

    style = style or PGAStyle()

    # Handle collection dimensions
    if blade.collection:
        _render_pga_points_batch(blade, canvas, style)
        return

    # Check if point or direction
    if is_point(blade):
        # Extract Euclidean coordinates
        coords = euclidean(blade)
        position = _to_3d(coords)
        canvas.point(position, color=style.color, radius=style.point_radius * style.scale)
    else:
        # Direction (point at infinity) - render as arrow from origin
        direction = bulk(blade)
        direction_3d = _to_3d(direction) * style.direction_arrow_length * style.scale
        canvas.arrow([0, 0, 0], direction_3d, color=style.color, shaft_radius=style.arrow_shaft_radius)


def _render_pga_points_batch(blade: Vector, canvas: Canvas, style: PGAStyle) -> None:
    """Render batch of PGA points."""
    cdim = len(blade.collection)
    n_points = blade.data.shape[0] if cdim == 1 else blade.data.reshape(-1, blade.dim).shape[0]

    for k in range(n_points):
        if cdim == 1:
            point_data = blade.data[k]
        else:
            point_data = blade.data.reshape(-1, blade.dim)[k]

        single_blade = Vector(data=point_data, grade=1, metric=blade.metric)

        if is_point(single_blade):
            coords = euclidean(single_blade)
            position = _to_3d(coords)
            canvas.point(position, color=style.color, radius=style.point_radius * style.scale)


def _to_3d(coords: NDArray) -> NDArray:
    """Convert coordinates to 3D (pad or truncate)."""
    coords = array(coords).flatten()
    result = zeros(3)
    n = min(len(coords), 3)
    result[:n] = coords[:n]
    return result


# =============================================================================
# PGA Line Rendering
# =============================================================================


def render_pga_line(
    blade: Vector,
    canvas: Canvas,
    style: PGAStyle | None = None,
) -> None:
    r"""
    Render PGA line (grade-2 in PGA) as extended line segment.

    A line in 3D PGA is represented as a bivector L = p /\ q for two points.
    We extract the direction and a point on the line to render it.
    """
    if blade.grade != 2:
        raise ValueError(f"render_pga_line requires grade-2, got {blade.grade}")

    style = style or PGAStyle()

    # Handle collection dimensions - render first
    if blade.collection:
        data = blade.data.reshape(-1, blade.dim, blade.dim)[0]
    else:
        data = blade.data

    # Extract line parameters from bivector
    # For PGA line L^{mn}, the direction is in the e_0 /\ e_k components
    # and the moment is in the e_k /\ e_l components

    dim = blade.dim
    if dim < 4:
        # 2D or less - not enough for line representation
        return

    # Direction: L^{0k} components give the direction
    direction = zeros(dim - 1)
    for k in range(1, dim):
        direction[k - 1] = data[0, k] - data[k, 0]

    dir_3d = _to_3d(direction)
    dir_norm = np_norm(dir_3d)

    if dir_norm < 1e-10:
        return

    dir_3d = dir_3d / dir_norm

    # Moment: L^{kl} for k,l > 0 gives position information
    # For visualization, find a point on the line using moment x direction
    moment = zeros(3)
    if dim >= 4:
        # In 3D PGA (dim=4), moment is (L^{23}, L^{31}, L^{12})
        moment[0] = data[2, 3] - data[3, 2] if dim > 3 else 0
        moment[1] = data[3, 1] - data[1, 3] if dim > 3 else 0
        moment[2] = data[1, 2] - data[2, 1] if dim > 2 else 0

    # Point on line: p = moment x direction (approximately)
    point_on_line = cross(moment, dir_3d)

    # Generate line segment
    extent = style.line_extent * style.scale
    start = point_on_line - dir_3d * extent
    end = point_on_line + dir_3d * extent

    points = array([start, end])
    canvas.curve(points, color=style.color, radius=style.line_radius * style.scale)


# =============================================================================
# PGA Plane Rendering
# =============================================================================


def render_pga_plane(
    blade: Vector,
    canvas: Canvas,
    style: PGAStyle | None = None,
) -> None:
    r"""
    Render PGA plane (grade-3 in PGA) as extended plane surface.

    A plane in 3D PGA is a trivector H = p /\ q /\ r for three points.
    """
    if blade.grade != 3:
        raise ValueError(f"render_pga_plane requires grade-3, got {blade.grade}")

    style = style or PGAStyle()

    # Handle collection dimensions - render first
    if blade.collection:
        data = blade.data.reshape(-1, blade.dim, blade.dim, blade.dim)[0]
    else:
        data = blade.data

    dim = blade.dim
    if dim < 4:
        return

    # Extract plane normal from trivector
    # The normal comes from the e_1 /\ e_2 /\ e_3 component (for 3D PGA)
    # and the distance from origin from e_0 components

    if dim >= 4:
        # Normal from components H^{0kl}
        normal = zeros(3)
        normal[0] = data[0, 2, 3] - data[0, 3, 2] + data[2, 3, 0] - data[3, 2, 0]  # e_1 direction
        normal[1] = data[0, 3, 1] - data[0, 1, 3] + data[3, 1, 0] - data[1, 3, 0]  # e_2 direction
        normal[2] = data[0, 1, 2] - data[0, 2, 1] + data[1, 2, 0] - data[2, 1, 0]  # e_3 direction

        norm_mag = np_norm(normal)
        if norm_mag < 1e-10:
            return
        normal = normal / norm_mag

        # Distance from origin (from H^{123} component)
        distance = (data[1, 2, 3] - data[1, 3, 2] + data[2, 3, 1] - data[2, 1, 3] + data[3, 1, 2] - data[3, 2, 1]) / 6

        center = normal * distance
    else:
        normal = array([0, 0, 1])
        center = array([0, 0, 0])

    canvas.plane(
        center,
        normal,
        size=style.plane_extent * style.scale,
        color=style.color,
        opacity=style.plane_opacity,
    )


# =============================================================================
# Context-Aware Visualization
# =============================================================================


def is_pga_context(vec: Vector) -> bool:
    """Check if blade has PGA context."""
    return vec.metric.structure == GAStructure.PROJECTIVE


def visualize_pga_blade(
    blade: Vector,
    canvas: Canvas | None = None,
    style: PGAStyle | None = None,
) -> Canvas:
    """
    Visualize a blade in PGA context with appropriate geometric interpretation.

    Args:
        blade: PGA blade (grade 1 = point, 2 = line, 3 = plane)
        canvas: Existing canvas or None
        style: PGA-specific style

    Returns:
        Canvas with rendered geometric object
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or PGAStyle()

    if blade.grade == 1:
        render_pga_point(blade, canvas, style)
    elif blade.grade == 2:
        render_pga_line(blade, canvas, style)
    elif blade.grade == 3:
        render_pga_plane(blade, canvas, style)
    else:
        # Grade 0 or grade 4+ in 3D PGA
        # For grade 4 (pseudoscalar), just mark origin
        canvas.point([0, 0, 0], color=style.color, radius=0.1)

    return canvas


def visualize_pga_scene(
    *blades: Vector,
    canvas: Canvas | None = None,
    style: PGAStyle | None = None,
) -> Canvas:
    """
    Visualize multiple PGA blades forming a geometric scene.

    Automatically colors blades using palette cycling.
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or PGAStyle()

    for blade in blades:
        # Create style with next palette color
        blade_style = PGAStyle(
            color=None,  # Will cycle
            **{k: v for k, v in style.__dict__.items() if k != "color"},
        )
        visualize_pga_blade(blade, canvas, blade_style)

    return canvas
