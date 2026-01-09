"""
Blade Drawing Functions

Core drawing functions for geometric algebra blades. Uses PyVista for rendering
with clean, professional aesthetics.

Key functions:
- draw_blade: Draw any blade (vector, bivector, trivector) at a position
- draw_coordinate_basis: Draw the standard basis vectors e1, e2, e3
"""

# Enable LaTeX/MathText rendering in VTK (must be before pyvista import)
try:
    import vtkmodules.vtkRenderingFreeType  # noqa: F401
    import vtkmodules.vtkRenderingMatplotlib  # noqa: F401
except ImportError:
    pass  # LaTeX rendering may not be available

import pyvista as pv
from numpy import abs as np_abs, argmax, array, cross, ndarray, zeros
from numpy.linalg import norm, svd

from morphis.ga.model import Blade
from morphis.visualization.theme import Color


# =============================================================================
# Blade Factorization
# =============================================================================


def _factor_bivector(B: Blade) -> tuple[ndarray, ndarray]:
    """
    Factor a bivector into two vectors: B = u ^ v.

    Uses SVD to find the two vectors that span the plane defined by the bivector.
    For the basis bivector e_i ^ e_j, returns (e_i, e_j).

    Returns:
        Tuple of two vectors (u, v) such that B = u ^ v (up to scale).
    """
    if B.grade != 2:
        raise ValueError(f"Expected grade 2, got {B.grade}")

    data = B.data
    dim = B.dim

    # The bivector B^{ab} is antisymmetric. We can find spanning vectors via SVD
    # of the matrix representation, or by finding two non-zero rows.
    # For simple blades, we find two vectors that span the plane.

    # Extract the antisymmetric matrix
    matrix = data.copy()

    # For numerical stability, use SVD
    U, S, Vt = svd(matrix)

    # The two largest singular values correspond to the plane
    # For a simple blade, only rank-2 structure exists
    if S[0] < 1e-10:
        # Zero bivector
        return zeros(dim), zeros(dim)

    # Scale by sqrt of singular value to distribute magnitude
    scale = S[0] ** 0.5
    u = U[:, 0] * scale
    v = Vt[0, :] * scale

    return u, v


def _factor_trivector(T: Blade) -> tuple[ndarray, ndarray, ndarray]:
    """
    Factor a trivector into three vectors: T = u ^ v ^ w.

    For basis trivector e_i ^ e_j ^ e_k, returns (e_i, e_j, e_k).

    Returns:
        Tuple of three vectors (u, v, w) such that T = u ^ v ^ w (up to scale).
    """
    if T.grade != 3:
        raise ValueError(f"Expected grade 3, got {T.grade}")

    data = T.data
    dim = T.dim

    # For 3D, the trivector T^{abc} = alpha * epsilon^{abc}
    # The spanning vectors are just e1, e2, e3 scaled appropriately

    # Find the magnitude from T^{012} (or any non-zero component)
    if dim >= 3:
        alpha = data[0, 1, 2]
        if np_abs(alpha) > 1e-10:
            # Standard orientation
            scale = np_abs(alpha) ** (1.0 / 3.0)
            u = array([scale, 0.0, 0.0])
            v = array([0.0, scale, 0.0])
            w = array([0.0, 0.0, scale])
            return u, v, w

    # Fallback: find spanning vectors from non-zero components
    # This is more complex for general trivectors
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                if np_abs(data[a, b, c]) > 1e-10:
                    # Found a non-zero component
                    scale = np_abs(data[a, b, c]) ** (1.0 / 3.0)
                    u = zeros(dim)
                    v = zeros(dim)
                    w = zeros(dim)
                    u[a] = scale
                    v[b] = scale
                    w[c] = scale
                    return u, v, w

    # Zero trivector
    return zeros(dim), zeros(dim), zeros(dim)


def factor_blade(B: Blade) -> tuple[ndarray, ...]:
    """
    Factor a blade into its constituent vectors.

    For a k-blade B = v1 ^ v2 ^ ... ^ vk, returns (v1, v2, ..., vk).

    Returns:
        Tuple of k vectors that wedge to produce the blade.
    """
    if B.grade == 0:
        return ()
    elif B.grade == 1:
        return (B.data.copy(),)
    elif B.grade == 2:
        return _factor_bivector(B)
    elif B.grade == 3:
        return _factor_trivector(B)
    else:
        raise NotImplementedError(f"Factorization not implemented for grade {B.grade}")


# =============================================================================
# Drawing Primitives
# =============================================================================


def _draw_arrow(
    plotter: pv.Plotter,
    start: ndarray,
    direction: ndarray,
    color: Color,
    shaft_radius: float = 0.008,
    tip_ratio: float = 0.12,
    tip_radius_ratio: float = 2.5,
    resolution: int = 20,
):
    """Draw an arrow from start in the given direction."""
    length = norm(direction)
    if length < 1e-10:
        return

    dir_norm = direction / length
    tip_length = length * tip_ratio
    shaft_length = length - tip_length
    tip_radius = shaft_radius * tip_radius_ratio

    shaft_end = start + dir_norm * shaft_length

    # Shaft cylinder
    shaft = pv.Cylinder(
        center=(start + shaft_end) / 2,
        direction=dir_norm,
        radius=shaft_radius,
        height=shaft_length,
        resolution=resolution,
        capping=True,
    )
    plotter.add_mesh(shaft, color=color, smooth_shading=True)

    # Tip cone
    tip = pv.Cone(
        center=shaft_end + dir_norm * (tip_length / 2),
        direction=dir_norm,
        height=tip_length,
        radius=tip_radius,
        resolution=resolution,
        capping=True,
    )
    plotter.add_mesh(tip, color=color, smooth_shading=True)


def _draw_edge(
    plotter: pv.Plotter,
    start: ndarray,
    end: ndarray,
    color: Color,
    radius: float = 0.006,
    resolution: int = 16,
):
    """Draw a tube edge between two points."""
    direction = end - start
    length = norm(direction)
    if length < 1e-10:
        return

    edge = pv.Spline(array([start, end]), n_points=2)
    tube = edge.tube(radius=radius)
    plotter.add_mesh(tube, color=color, smooth_shading=True)


def _draw_parallelogram(
    plotter: pv.Plotter,
    origin: ndarray,
    u: ndarray,
    v: ndarray,
    color: Color,
    tetrad: bool = True,
    surface: bool = True,
    edge_radius: float = 0.006,
    opacity: float = 0.25,
):
    """Draw a parallelogram defined by vectors u and v from origin."""
    corners = [
        origin,
        origin + u,
        origin + u + v,
        origin + v,
    ]

    if tetrad:
        # Draw edges
        for i in range(4):
            _draw_edge(plotter, corners[i], corners[(i + 1) % 4], color, edge_radius)

    if surface:
        # Draw filled quad
        quad = pv.Quadrilateral(corners)
        plotter.add_mesh(quad, color=color, opacity=opacity, smooth_shading=True)


def _draw_parallelepiped(
    plotter: pv.Plotter,
    origin: ndarray,
    u: ndarray,
    v: ndarray,
    w: ndarray,
    color: Color,
    tetrad: bool = True,
    surface: bool = True,
    edge_radius: float = 0.006,
    opacity: float = 0.15,
):
    """Draw a parallelepiped defined by vectors u, v, w from origin."""
    # 8 corners
    corners = [
        origin,  # 0
        origin + u,  # 1
        origin + v,  # 2
        origin + w,  # 3
        origin + u + v,  # 4
        origin + u + w,  # 5
        origin + v + w,  # 6
        origin + u + v + w,  # 7
    ]

    if tetrad:
        # 12 edges
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        for i, j in edges:
            _draw_edge(plotter, corners[i], corners[j], color, edge_radius)

    if surface:
        # 6 faces
        faces = [
            [0, 1, 4, 2],  # bottom
            [3, 5, 7, 6],  # top
            [0, 1, 5, 3],  # front
            [2, 4, 7, 6],  # back
            [0, 2, 6, 3],  # left
            [1, 4, 7, 5],  # right
        ]
        for face_indices in faces:
            quad = pv.Quadrilateral([corners[i] for i in face_indices])
            plotter.add_mesh(quad, color=color, opacity=opacity, smooth_shading=True)


# =============================================================================
# Main Drawing Functions
# =============================================================================


def draw_blade(
    plotter: pv.Plotter,
    b: Blade,
    p: Blade | ndarray | tuple = None,
    color: Color = (0.85, 0.85, 0.85),
    tetrad: bool = True,
    surface: bool = True,
    label: bool = False,
    name: str | None = None,
    shaft_radius: float = 0.008,
    edge_radius: float = 0.006,
    label_offset: float = 0.08,
):
    """
    Draw a blade at a given position.

    For vectors: draws an arrow
    For bivectors: draws a parallelogram (edges and/or surface)
    For trivectors: draws a parallelepiped (edges and/or surface)

    Args:
        plotter: PyVista plotter
        b: The blade to draw
        p: Position (origin) for the blade. Can be Blade, array, or tuple.
           Defaults to origin.
        color: RGB color tuple (0-1 range)
        tetrad: If True, draw edges/arrows
        surface: If True, draw filled surfaces (for grade >= 2)
        label: If True, add a text label
        name: Custom label text. If None, auto-generates based on blade
              (e.g., "e1" for basis vectors, "e12" for basis bivectors)
        shaft_radius: Radius for arrow shafts
        edge_radius: Radius for edge tubes
        label_offset: Distance to offset labels from geometry
    """
    # Handle position
    if p is None:
        origin = zeros(3)
    elif isinstance(p, Blade):
        origin = p.data[:3] if len(p.data) >= 3 else array([*p.data, *[0.0] * (3 - len(p.data))])
    elif isinstance(p, (list, tuple)):
        origin = array(p, dtype=float)
    else:
        origin = array(p, dtype=float)

    # Ensure 3D
    if len(origin) < 3:
        origin = array([*origin, *[0.0] * (3 - len(origin))])
    origin = origin[:3]

    grade = b.grade

    if grade == 0:
        # Scalar: draw a point
        sphere = pv.Sphere(radius=0.02, center=origin)
        plotter.add_mesh(sphere, color=color, smooth_shading=True)

    elif grade == 1:
        # Vector: draw arrow
        direction = b.data[:3] if len(b.data) >= 3 else array([*b.data, *[0.0] * (3 - len(b.data))])
        if tetrad:
            _draw_arrow(plotter, origin, direction, color, shaft_radius=shaft_radius)

    elif grade == 2:
        # Bivector: factor and draw parallelogram
        u, v = _factor_bivector(b)
        u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
        v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
        _draw_parallelogram(plotter, origin, u, v, color, tetrad=tetrad, surface=surface, edge_radius=edge_radius)

    elif grade == 3:
        # Trivector: factor and draw parallelepiped
        u, v, w = _factor_trivector(b)
        u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
        v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
        w = w[:3] if len(w) >= 3 else array([*w, *[0.0] * (3 - len(w))])
        _draw_parallelepiped(plotter, origin, u, v, w, color, tetrad=tetrad, surface=surface, edge_radius=edge_radius)

    else:
        raise NotImplementedError(f"Drawing not implemented for grade {grade}")

    if label:
        # Generate label position and text
        if grade == 1:
            direction = b.data[:3] if len(b.data) >= 3 else array([*b.data, *[0.0] * (3 - len(b.data))])
            # Find dominant axis for default label
            if name is None:
                dominant = int(argmax(np_abs(direction))) + 1
                name = f"e{dominant}"
            # Offset perpendicular to direction
            label_pos = origin + direction * 0.5 + array([0, -1, -1]) / norm(array([0, 1, 1])) * label_offset
        elif grade == 2:
            u, v = _factor_bivector(b)
            u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
            v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
            # Default label from dominant axes
            if name is None:
                idx1 = int(argmax(np_abs(u))) + 1
                idx2 = int(argmax(np_abs(v))) + 1
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                name = f"e{idx1}{idx2}"
            normal = cross(u, v)
            if norm(normal) > 1e-10:
                normal = normal / norm(normal)
            label_dir = -normal if normal[2] >= 0 else normal
            label_pos = origin + (u + v) / 2 + label_dir * label_offset * 2
        elif grade == 3:
            u, v, w = _factor_trivector(b)
            u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
            v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
            w = w[:3] if len(w) >= 3 else array([*w, *[0.0] * (3 - len(w))])
            if name is None:
                name = "e123"
            label_pos = origin + (u + v + w) / 2 - array([1, 1, 1]) / norm(array([1, 1, 1])) * label_offset * 3
        else:
            label_pos = origin
            if name is None:
                name = f"grade-{grade}"

        plotter.add_point_labels(
            [label_pos],
            [name],
            font_size=12,
            text_color=color,
            point_size=0,
            shape=None,
            show_points=False,
            always_visible=True,
        )


def draw_coordinate_basis(
    plotter: pv.Plotter,
    scale: float = 1.0,
    color: Color = (0.85, 0.85, 0.85),
    tetrad: bool = True,
    surface: bool = False,
    label: bool = True,
    label_offset: float = 0.08,
):
    """
    Draw the standard coordinate basis e1, e2, e3.

    This is a convenience wrapper around draw_blade for the common case
    of drawing orthonormal basis vectors.

    Args:
        plotter: PyVista plotter
        scale: Length of basis vectors
        color: RGB color tuple (0-1 range)
        tetrad: If True, draw arrows
        surface: If True, draw filled surfaces (not typical for vectors)
        label: If True, add e1, e2, e3 labels
        label_offset: Distance to offset labels from axes
    """
    directions = [
        array([1.0, 0.0, 0.0]) * scale,
        array([0.0, 1.0, 0.0]) * scale,
        array([0.0, 0.0, 1.0]) * scale,
    ]

    # Label offset directions (outside positive octant)
    offset_dirs = [
        array([0, -1, -1]),  # e1: offset in -y-z
        array([-1, 0, -1]),  # e2: offset in -x-z
        array([-1, -1, 0]),  # e3: offset in -x-y
    ]

    names = [r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_3$"]

    for direction, offset_dir, axis_name in zip(directions, offset_dirs, names, strict=False):
        # Draw the arrow
        if tetrad:
            _draw_arrow(plotter, zeros(3), direction, color, shaft_radius=0.004, tip_ratio=0.08)

        # Add label
        if label:
            offset = offset_dir / norm(offset_dir) * label_offset
            label_pos = direction * 0.5 + offset
            plotter.add_point_labels(
                [label_pos],
                [axis_name],
                font_size=12,
                text_color=color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )


def draw_basis_blade(
    plotter: pv.Plotter,
    indices: tuple[int, ...],
    position: ndarray | tuple = (0, 0, 0),
    scale: float = 1.0,
    color: Color = (0.85, 0.85, 0.85),
    tetrad: bool = True,
    surface: bool = True,
    label: bool = False,
    name: str | None = None,
    label_offset: float = 0.08,
):
    """
    Draw a basis k-blade (e1, e12, e123, etc.) at a given position.

    This creates a blade from basis indices and draws it.

    Args:
        plotter: PyVista plotter
        indices: Tuple of basis indices (1-indexed), e.g., (1,), (1, 2), (1, 2, 3)
        position: Position to draw the blade
        scale: Size scaling factor
        color: RGB color tuple
        tetrad: Draw edges/arrows
        surface: Draw filled surfaces
        label: If True, show label
        name: Custom label text. If None, auto-generates (e.g., "e12" for indices (1,2))
        label_offset: Distance to offset label

    Examples:
        draw_basis_blade(plotter, (1,))       # e1 vector
        draw_basis_blade(plotter, (1, 2))     # e12 bivector
        draw_basis_blade(plotter, (1, 2, 3))  # e123 trivector
    """
    position = array(position, dtype=float)
    grade = len(indices)

    # Generate default name from indices
    if name is None:
        name = "e" + "".join(str(i) for i in indices)

    # Map indices to directions (1-indexed to 0-indexed)
    directions = {1: array([1, 0, 0]), 2: array([0, 1, 0]), 3: array([0, 0, 1])}

    if grade == 1:
        # Vector
        idx = indices[0]
        direction = directions.get(idx, array([1, 0, 0])) * scale

        if tetrad:
            _draw_arrow(plotter, position, direction, color, shaft_radius=0.004, tip_ratio=0.08)

        if label:
            offset_dirs = {
                1: array([0, -1, -1]),
                2: array([-1, 0, -1]),
                3: array([-1, -1, 0]),
            }
            offset_dir = offset_dirs.get(idx, array([0, -1, 0]))
            offset = offset_dir / norm(offset_dir) * label_offset
            label_pos = position + direction * 0.5 + offset
            plotter.add_point_labels(
                [label_pos],
                [name],
                font_size=12,
                text_color=color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )

    elif grade == 2:
        # Bivector
        u = directions.get(indices[0], array([1, 0, 0])) * scale
        v = directions.get(indices[1], array([0, 1, 0])) * scale

        _draw_parallelogram(plotter, position, u, v, color, tetrad=tetrad, surface=surface)

        if label:
            normal = cross(u, v)
            if norm(normal) > 1e-10:
                normal = normal / norm(normal)
            label_dir = -normal if normal[2] >= 0 else normal
            label_pos = position + (u + v) / 2 + label_dir * label_offset * 2
            plotter.add_point_labels(
                [label_pos],
                [name],
                font_size=12,
                text_color=color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )

    elif grade == 3:
        # Trivector
        u = directions.get(indices[0], array([1, 0, 0])) * scale
        v = directions.get(indices[1], array([0, 1, 0])) * scale
        w = directions.get(indices[2], array([0, 0, 1])) * scale

        _draw_parallelepiped(plotter, position, u, v, w, color, tetrad=tetrad, surface=surface)

        if label:
            label_pos = position + (u + v + w) / 2 + array([-1, -1, -1]) / norm(array([1, 1, 1])) * label_offset * 3
            plotter.add_point_labels(
                [label_pos],
                [name],
                font_size=12,
                text_color=color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )

    else:
        raise NotImplementedError(f"Drawing not implemented for grade {grade}")
