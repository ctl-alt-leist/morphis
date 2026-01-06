"""
Blade Visualization

Rendering functions for geometric algebra blades of all grades. Handles
dimensional projection for d > 3, context-aware rendering (PGA, etc.),
and multiple visualization modes per grade.

Each blade renders to canvas primitives (arrows, curves, points, planes).
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from numpy import (
    array,
    cos,
    cross,
    linspace,
    pi,
    sign,
    sin,
    sqrt,
    stack,
    zeros,
)
from numpy.linalg import norm as np_norm, svd
from numpy.typing import NDArray

from morphis.ga.model import Blade
from morphis.visualization.canvas import Canvas
from morphis.visualization.projection import ProjectionConfig, project_blade
from morphis.visualization.theme import Color


# =============================================================================
# Style Configuration
# =============================================================================


@dataclass
class BladeStyle:
    """Style parameters for blade rendering."""

    color: Optional[Color] = None
    opacity: float = 0.8
    scale: float = 1.0

    # Scalar specific
    scalar_radius_base: float = 0.05
    scalar_radius_scale: float = 0.03
    positive_color: Optional[Color] = None
    negative_color: Optional[Color] = None

    # Vector specific
    arrow_shaft_radius: Optional[float] = None
    arrow_origin: Optional[Tuple[float, float, float]] = None

    # Bivector specific
    bivector_radius_factor: float = 0.3
    bivector_resolution: int = 48
    parallelogram_opacity: float = 0.4

    # Trivector specific
    volume_opacity: float = 0.25
    edge_radius: float = 0.006

    # Plane rendering
    plane_size: float = 1.0
    plane_opacity: float = 0.25

    # Dual rendering
    dual_color: Optional[Color] = None
    dual_opacity: float = 0.5


# =============================================================================
# Scalar Rendering (Grade 0)
# =============================================================================


def render_scalar(
    blade: Blade,
    canvas: Canvas,
    position: Optional[Tuple[float, float, float]] = None,
    style: Optional[BladeStyle] = None,
) -> None:
    """
    Render scalar blade as sphere at origin (or specified position).

    Magnitude maps to sphere radius. Sign shown by color:
    - Positive: uses positive_color or theme palette
    - Negative: uses negative_color or theme e1 (red family)
    """
    if blade.grade != 0:
        raise ValueError(f"render_scalar requires grade-0, got {blade.grade}")

    style = style or BladeStyle()
    position = position or (0.0, 0.0, 0.0)

    # Handle collection dimensions - render first element for now
    scalar_value = float(blade.data.flat[0]) if blade.cdim > 0 else float(blade.data)

    magnitude = abs(scalar_value)
    radius = style.scalar_radius_base + magnitude * style.scalar_radius_scale * style.scale

    if scalar_value >= 0:
        color = style.positive_color or style.color
    else:
        color = style.negative_color or canvas.theme.e1

    canvas.point(position, color=color, radius=radius)


# =============================================================================
# Vector Rendering (Grade 1)
# =============================================================================


def render_vector(
    blade: Blade,
    canvas: Canvas,
    projection: Optional[ProjectionConfig] = None,
    style: Optional[BladeStyle] = None,
) -> None:
    """
    Render vector blade as arrow from origin.

    For d > 3: Projects to 3D using specified or default projection.
    Handles collection dimensions by rendering multiple arrows.
    """
    if blade.grade != 1:
        raise ValueError(f"render_vector requires grade-1, got {blade.grade}")

    style = style or BladeStyle()

    # Project if needed
    if blade.dim > 3:
        blade = project_blade(blade, projection or ProjectionConfig())

    origin = array(style.arrow_origin) if style.arrow_origin else array([0.0, 0.0, 0.0])

    # Handle collection dimensions
    if blade.cdim == 0:
        # Single vector
        direction = blade.data * style.scale
        if blade.dim < 3:
            direction = _pad_to_3d(direction)
        canvas.arrow(origin, direction, color=style.color, shaft_radius=style.arrow_shaft_radius)
    else:
        # Collection of vectors
        flat_data = blade.data.reshape(-1, blade.dim)
        n_vectors = flat_data.shape[0]
        origins = stack([origin] * n_vectors)
        directions = flat_data * style.scale

        if blade.dim < 3:
            directions = stack([_pad_to_3d(d) for d in directions])

        canvas.arrows(origins, directions, color=style.color, shaft_radius=style.arrow_shaft_radius)


def _pad_to_3d(vec: NDArray) -> NDArray:
    """Pad a vector to 3D by adding zeros."""
    result = zeros(3)
    result[: len(vec)] = vec
    return result


# =============================================================================
# Bivector Rendering (Grade 2)
# =============================================================================


def render_bivector(
    blade: Blade,
    canvas: Canvas,
    mode: Literal["circle", "parallelogram", "plane", "circular_arrow"] = "circle",
    projection: Optional[ProjectionConfig] = None,
    style: Optional[BladeStyle] = None,
) -> None:
    """
    Render bivector blade with multiple visualization modes.

    Modes:
        'circle': Circle in the plane, radius proportional to magnitude
        'parallelogram': Spanning parallelogram from decomposed vectors
        'plane': Semi-transparent plane with normal arrow
        'circular_arrow': Circle with orientation arrow

    For d > 3: Projects to 3D first.
    """
    if blade.grade != 2:
        raise ValueError(f"render_bivector requires grade-2, got {blade.grade}")

    style = style or BladeStyle()

    # Project if needed
    if blade.dim > 3:
        blade = project_blade(blade, projection or ProjectionConfig())

    # Handle collection dimensions - render first element
    if blade.cdim > 0:
        data = blade.data.reshape(-1, blade.dim, blade.dim)[0]
    else:
        data = blade.data

    # Pad to 3D if needed
    if blade.dim < 3:
        padded = zeros((3, 3))
        padded[: blade.dim, : blade.dim] = data
        data = padded

    if mode == "circle":
        _render_bivector_circle(data, canvas, style)
    elif mode == "parallelogram":
        _render_bivector_parallelogram(data, canvas, style)
    elif mode == "plane":
        _render_bivector_plane(data, canvas, style)
    elif mode == "circular_arrow":
        _render_bivector_circular_arrow(data, canvas, style)
    else:
        raise ValueError(f"Unknown bivector mode: {mode}")


def _bivector_to_normal_and_magnitude(B: NDArray) -> Tuple[NDArray, float]:
    """
    Extract normal vector and magnitude from 3D bivector.

    For B^{mn}, the normal is n_k = (1/2) eps_{kmn} B^{mn}.
    Magnitude is sqrt(sum of squared antisymmetric components).
    """
    # Extract antisymmetric components
    B_01 = (B[0, 1] - B[1, 0]) / 2
    B_02 = (B[0, 2] - B[2, 0]) / 2
    B_12 = (B[1, 2] - B[2, 1]) / 2

    # Normal from Hodge dual: n = (B_12, -B_02, B_01)
    normal = array([B_12, -B_02, B_01])
    magnitude = sqrt(B_01**2 + B_02**2 + B_12**2)

    if magnitude > 1e-10:
        normal = normal / magnitude

    return normal, magnitude


def _bivector_to_spanning_vectors(B: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Decompose bivector into two spanning vectors a, b such that B ~ a ∧ b.

    Uses SVD to find the principal directions.
    """
    # Make antisymmetric
    B_anti = (B - B.T) / 2

    # SVD gives us the principal plane
    U, S, Vt = svd(B_anti)

    # The two largest singular values correspond to the plane
    # For antisymmetric matrices, singular values come in pairs
    if S[0] > 1e-10:
        magnitude = sqrt(S[0])
        # Extract spanning vectors from U
        a = U[:, 0] * magnitude
        b = U[:, 1] * magnitude
    else:
        # Degenerate case
        a = array([1.0, 0.0, 0.0])
        b = array([0.0, 1.0, 0.0])

    return a, b


def _orthonormal_basis_in_plane(normal: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Construct orthonormal basis vectors in plane with given normal.
    """
    normal = normal / (np_norm(normal) + 1e-12)

    # Choose a vector not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = array([1.0, 0.0, 0.0])
    else:
        ref = array([0.0, 1.0, 0.0])

    u = cross(normal, ref)
    u = u / (np_norm(u) + 1e-12)
    v = cross(normal, u)

    return u, v


def _render_bivector_circle(B: NDArray, canvas: Canvas, style: BladeStyle) -> None:
    """Render bivector as circle in the plane."""
    normal, magnitude = _bivector_to_normal_and_magnitude(B)

    if magnitude < 1e-10:
        return

    radius = sqrt(magnitude) * style.bivector_radius_factor * style.scale
    u, v = _orthonormal_basis_in_plane(normal)

    # Generate circle points
    theta = linspace(0, 2 * pi, style.bivector_resolution)
    points = stack([radius * (cos(t) * u + sin(t) * v) for t in theta])

    canvas.curve(points, color=style.color, radius=style.edge_radius)


def _render_bivector_parallelogram(B: NDArray, canvas: Canvas, style: BladeStyle) -> None:
    """Render bivector as parallelogram from spanning vectors."""
    a, b = _bivector_to_spanning_vectors(B)

    a = a * style.scale
    b = b * style.scale

    origin = array([0.0, 0.0, 0.0])

    # Draw the two spanning vectors as arrows
    canvas.arrow(origin, a, color=style.color, shaft_radius=style.arrow_shaft_radius)
    canvas.arrow(origin, b, color=style.color, shaft_radius=style.arrow_shaft_radius)

    # Draw parallelogram edges
    corners = array([
        origin,
        a,
        a + b,
        b,
        origin,
    ])
    canvas.curve(corners, color=style.color, radius=style.edge_radius)

    # Semi-transparent fill
    normal = cross(a, b)
    if np_norm(normal) > 1e-10:
        normal = normal / np_norm(normal)
        center = (a + b) / 2
        size = max(np_norm(a), np_norm(b)) * 1.5
        canvas.plane(center, normal, size=size, color=style.color, opacity=style.parallelogram_opacity)


def _render_bivector_plane(B: NDArray, canvas: Canvas, style: BladeStyle) -> None:
    """Render bivector as semi-transparent plane with normal arrow."""
    normal, magnitude = _bivector_to_normal_and_magnitude(B)

    if magnitude < 1e-10:
        return

    normal_length = sqrt(magnitude) * style.scale * 0.5
    center = array([0.0, 0.0, 0.0])

    # Draw the plane
    canvas.plane(
        center,
        normal,
        size=style.plane_size * style.scale,
        color=style.color,
        opacity=style.plane_opacity,
    )

    # Draw normal vector
    canvas.arrow(
        center,
        normal * normal_length,
        color=style.color,
        shaft_radius=style.arrow_shaft_radius,
    )


def _render_bivector_circular_arrow(B: NDArray, canvas: Canvas, style: BladeStyle) -> None:
    """Render bivector as circle with orientation arrow."""
    normal, magnitude = _bivector_to_normal_and_magnitude(B)

    if magnitude < 1e-10:
        return

    radius = sqrt(magnitude) * style.bivector_radius_factor * style.scale
    u, v = _orthonormal_basis_in_plane(normal)

    # Generate partial circle (leave gap for arrow)
    n_points = style.bivector_resolution
    theta = linspace(0, 1.85 * pi, n_points)
    points = stack([radius * (cos(t) * u + sin(t) * v) for t in theta])

    canvas.curve(points, color=style.color, radius=style.edge_radius)

    # Add arrow head at end of circle
    end_point = points[-1]
    # Tangent direction at end
    tangent = -sin(theta[-1]) * u + cos(theta[-1]) * v
    arrow_length = radius * 0.25
    canvas.arrow(
        end_point - tangent * arrow_length * 0.5,
        tangent * arrow_length,
        color=style.color,
        shaft_radius=style.edge_radius * 1.5,
    )


# =============================================================================
# Trivector Rendering (Grade 3)
# =============================================================================


def render_trivector(
    blade: Blade,
    canvas: Canvas,
    mode: Literal["parallelepiped", "sphere"] = "parallelepiped",
    projection: Optional[ProjectionConfig] = None,
    style: Optional[BladeStyle] = None,
) -> None:
    """
    Render trivector blade as 3D volume element.

    Modes:
        'parallelepiped': Volume from three spanning vectors
        'sphere': Sphere with radius proportional to magnitude^(1/3)

    For d > 3: Projects to 3D first.
    """
    if blade.grade != 3:
        raise ValueError(f"render_trivector requires grade-3, got {blade.grade}")

    style = style or BladeStyle()

    # Project if needed
    if blade.dim > 3:
        blade = project_blade(blade, projection or ProjectionConfig())

    # Handle collection dimensions - render first element
    if blade.cdim > 0:
        data = blade.data.reshape(-1, blade.dim, blade.dim, blade.dim)[0]
    else:
        data = blade.data

    if mode == "parallelepiped":
        _render_trivector_parallelepiped(data, canvas, style)
    elif mode == "sphere":
        _render_trivector_sphere(data, canvas, style)
    else:
        raise ValueError(f"Unknown trivector mode: {mode}")


def _trivector_magnitude(T: NDArray) -> float:
    """
    Compute magnitude of trivector from its components.

    For 3D, T^{012} is the only independent component.
    """
    # Extract the antisymmetric part
    # T^{012} = T[0,1,2] - T[0,2,1] + T[1,2,0] - T[1,0,2] + T[2,0,1] - T[2,1,0]
    T_012 = (T[0, 1, 2] - T[0, 2, 1] + T[1, 2, 0] - T[1, 0, 2] + T[2, 0, 1] - T[2, 1, 0]) / 6

    return abs(T_012)


def _trivector_to_spanning_vectors(T: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Decompose trivector into three spanning vectors.

    For visualization, we use the magnitude to scale canonical basis.
    """
    magnitude = _trivector_magnitude(T)

    if magnitude < 1e-10:
        scale = 0.1
    else:
        scale = magnitude ** (1 / 3)

    # Determine orientation from sign
    T_012 = (T[0, 1, 2] - T[0, 2, 1] + T[1, 2, 0] - T[1, 0, 2] + T[2, 0, 1] - T[2, 1, 0]) / 6

    orientation = sign(T_012) if abs(T_012) > 1e-10 else 1.0

    a = array([scale, 0.0, 0.0])
    b = array([0.0, scale, 0.0])
    c = array([0.0, 0.0, scale * orientation])

    return a, b, c


def _render_trivector_parallelepiped(T: NDArray, canvas: Canvas, style: BladeStyle) -> None:
    """Render trivector as parallelepiped from spanning vectors."""
    a, b, c = _trivector_to_spanning_vectors(T)

    a = a * style.scale
    b = b * style.scale
    c = c * style.scale

    origin = array([0.0, 0.0, 0.0])

    # 8 corners
    corners = [
        origin,
        a,
        b,
        c,
        a + b,
        a + c,
        b + c,
        a + b + c,
    ]

    # 12 edges
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),  # From origin
        (1, 4),
        (1, 5),  # From a
        (2, 4),
        (2, 6),  # From b
        (3, 5),
        (3, 6),  # From c
        (4, 7),
        (5, 7),
        (6, 7),  # To far corner
    ]

    for start_idx, end_idx in edges:
        edge_points = array([corners[start_idx], corners[end_idx]])
        canvas.curve(edge_points, color=style.color, radius=style.edge_radius)

    # Draw spanning vectors as arrows
    canvas.arrow(origin, a, color=style.color, shaft_radius=style.arrow_shaft_radius)
    canvas.arrow(origin, b, color=style.color, shaft_radius=style.arrow_shaft_radius)
    canvas.arrow(origin, c, color=style.color, shaft_radius=style.arrow_shaft_radius)

    # Semi-transparent faces (just the three at origin)
    canvas.plane(a / 2, cross(b, c), size=np_norm(b), color=style.color, opacity=style.volume_opacity)
    canvas.plane(b / 2, cross(c, a), size=np_norm(c), color=style.color, opacity=style.volume_opacity)
    canvas.plane(c / 2, cross(a, b), size=np_norm(a), color=style.color, opacity=style.volume_opacity)


def _render_trivector_sphere(T: NDArray, canvas: Canvas, style: BladeStyle) -> None:
    """Render trivector as sphere with radius proportional to magnitude^(1/3)."""
    magnitude = _trivector_magnitude(T)

    if magnitude < 1e-10:
        return

    radius = (magnitude ** (1 / 3)) * 0.3 * style.scale
    canvas.point([0, 0, 0], color=style.color, radius=radius)


# =============================================================================
# Main Visualization API
# =============================================================================


def visualize_blade(
    blade: Blade,
    canvas: Optional[Canvas] = None,
    mode: str = "auto",
    projection: Optional[ProjectionConfig] = None,
    show_dual: bool = False,
    style: Optional[BladeStyle] = None,
) -> Canvas:
    """
    Main entry point for blade visualization.

    Args:
        blade: Blade to visualize
        canvas: Existing canvas (creates new if None)
        mode: Rendering mode (grade-specific, or 'auto' to choose)
        projection: Configuration for d > 3 projection
        show_dual: Also render the dual blade
        style: Style parameters

    Returns:
        Canvas with rendered blade(s)
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or BladeStyle()

    # Determine mode based on grade if auto
    if mode == "auto":
        mode = _auto_mode(blade)

    # Render based on grade
    if blade.grade == 0:
        render_scalar(blade, canvas, style=style)
    elif blade.grade == 1:
        render_vector(blade, canvas, projection=projection, style=style)
    elif blade.grade == 2:
        render_bivector(blade, canvas, mode=mode, projection=projection, style=style)
    elif blade.grade == 3:
        render_trivector(blade, canvas, mode=mode, projection=projection, style=style)
    else:
        # Higher grades: try to project or use dual
        _render_higher_grade(blade, canvas, projection, style, show_dual)

    # Render dual if requested
    if show_dual and blade.grade > 0 and blade.grade < blade.dim:
        _render_dual(blade, canvas, projection, style)

    return canvas


def visualize_blades(
    blades: list,
    canvas: Optional[Canvas] = None,
    style: Optional[BladeStyle] = None,
) -> Canvas:
    """
    Visualize multiple blades on same canvas.

    Uses canvas color cycling automatically.
    """
    if canvas is None:
        canvas = Canvas(show_basis=True)

    style = style or BladeStyle()

    for blade in blades:
        # Use next palette color for each blade
        blade_style = BladeStyle(
            color=None,  # Will use canvas cycling
            **{k: v for k, v in style.__dict__.items() if k != "color"},
        )
        visualize_blade(blade, canvas, style=blade_style)

    return canvas


def _auto_mode(blade: Blade) -> str:
    """Select default rendering mode based on grade."""
    if blade.grade == 2:
        return "circle"
    elif blade.grade == 3:
        return "parallelepiped"
    return "auto"


def _render_higher_grade(
    blade: Blade,
    canvas: Canvas,
    projection: Optional[ProjectionConfig],
    style: BladeStyle,
    show_dual: bool,
) -> None:
    """Render blades with grade > 3."""
    from morphis.ga.duality import right_complement

    # For grade > 3, render the dual which has grade d - k
    dual = right_complement(blade)
    dual_grade = dual.grade

    if dual_grade <= 3:
        dual_style = BladeStyle(
            color=style.dual_color or style.color,
            opacity=style.dual_opacity,
            scale=style.scale,
        )
        visualize_blade(dual, canvas, mode="auto", projection=projection, style=dual_style)
    else:
        # Both original and dual are high grade - just show a marker
        canvas.point([0, 0, 0], color=style.color, radius=0.1)


def _render_dual(
    blade: Blade,
    canvas: Canvas,
    projection: Optional[ProjectionConfig],
    style: BladeStyle,
) -> None:
    """Render the dual of a blade."""
    from morphis.ga.duality import right_complement

    dual = right_complement(blade)

    dual_style = BladeStyle(
        color=style.dual_color or canvas.theme.muted,
        opacity=style.dual_opacity,
        scale=style.scale * 0.8,
    )

    visualize_blade(dual, canvas, mode="auto", projection=projection, style=dual_style)
