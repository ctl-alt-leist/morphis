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
from numpy.linalg import norm

from morphis.geometry.algebra.factorization import spanning_vectors
from morphis.geometry.model import Blade
from morphis.visualization.theme import Color


def _extract_vector_data(blade: Blade) -> ndarray:
    """Extract the raw numpy array from a grade-1 blade."""
    return blade.data


def _factor_blade_to_arrays(B: Blade) -> tuple[ndarray, ...]:
    """
    Factor a blade and return raw numpy arrays (for mesh creation).

    This is an internal helper that wraps the GA factorization for visualization.
    """
    vectors = spanning_vectors(B)
    return tuple(_extract_vector_data(v) for v in vectors)


# =============================================================================
# Mesh Creation (low-level, returns meshes without adding to plotter)
# =============================================================================


def _create_arrow_mesh(
    start: ndarray,
    direction: ndarray,
    shaft_radius: float = 0.008,
    tip_ratio: float = 0.12,
    tip_radius_ratio: float = 2.5,
    resolution: int = 20,
) -> pv.PolyData | None:
    """Create an arrow mesh from start in the given direction."""
    length = norm(direction)
    if length < 1e-10:
        return None

    dir_norm = direction / length
    tip_length = length * tip_ratio
    shaft_length = length - tip_length
    tip_radius = shaft_radius * tip_radius_ratio

    shaft_end = start + dir_norm * shaft_length

    shaft = pv.Cylinder(
        center=(start + shaft_end) / 2,
        direction=dir_norm,
        radius=shaft_radius,
        height=shaft_length,
        resolution=resolution,
        capping=True,
    )

    tip = pv.Cone(
        center=shaft_end + dir_norm * (tip_length / 2),
        direction=dir_norm,
        height=tip_length,
        radius=tip_radius,
        resolution=resolution,
        capping=True,
    )

    return shaft.merge(tip)


def _create_origin_marker(origin: ndarray, radius: float = 0.025) -> pv.PolyData:
    """Create a sphere mesh to mark the origin point."""
    return pv.Sphere(radius=radius, center=origin)


def create_vector_mesh(
    origin: ndarray,
    direction: ndarray,
    shaft_radius: float = 0.008,
) -> tuple[pv.PolyData | None, pv.PolyData]:
    """
    Create meshes for a vector (grade 1 blade).

    Returns:
        (arrow_mesh, origin_marker_mesh)
    """
    arrow = _create_arrow_mesh(origin, direction, shaft_radius=shaft_radius)
    marker = _create_origin_marker(origin)
    return arrow, marker


def create_bivector_mesh(
    origin: ndarray,
    u: ndarray,
    v: ndarray,
    shaft_radius: float = 0.006,
    face_opacity: float = 0.25,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """
    Create meshes for a bivector (grade 2 blade).

    Bivector is an OPEN surface - it has a boundary.
    Show boundary circulation: u → v → -u → -v

    Returns:
        (edges_mesh, face_mesh, origin_marker_mesh)
    """
    # Boundary circulation arrows (open surface has a boundary)
    boundary_arrows = [
        (origin, u),  # along +u
        (origin + u, v),  # along +v
        (origin + u + v, -u),  # along -u
        (origin + v, -v),  # along -v (back to origin)
    ]

    edge_meshes = []
    for start, direction in boundary_arrows:
        arrow = _create_arrow_mesh(start, direction, shaft_radius=shaft_radius)
        if arrow is not None:
            edge_meshes.append(arrow)

    edges_mesh = edge_meshes[0] if edge_meshes else pv.PolyData()
    for mesh in edge_meshes[1:]:
        edges_mesh = edges_mesh.merge(mesh)

    # Face
    corners = [origin, origin + u, origin + u + v, origin + v]
    face_mesh = pv.Quadrilateral(corners)

    # Origin marker
    marker = _create_origin_marker(origin)

    return edges_mesh, face_mesh, marker


def _create_tube_mesh(
    start: ndarray,
    end: ndarray,
    radius: float = 0.006,
    resolution: int = 20,
) -> pv.PolyData | None:
    """Create a tube (cylinder) mesh between two points."""
    direction = end - start
    length = norm(direction)
    if length < 1e-10:
        return None

    dir_norm = direction / length
    center = (start + end) / 2

    return pv.Cylinder(
        center=center,
        direction=dir_norm,
        radius=radius,
        height=length,
        resolution=resolution,
        capping=True,
    )


def create_trivector_mesh(
    origin: ndarray,
    u: ndarray,
    v: ndarray,
    w: ndarray,
    shaft_radius: float = 0.006,
    face_opacity: float = 0.15,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """
    Create meshes for a trivector.

    Trivector is a CLOSED surface - ∂∂=0, no consistent boundary orientation.
    Show orientation via path: 0 → u → u+v → u+v+w (3 arrows along cumulative sum).
    All other edges are tubes.

    Returns:
        (edges_mesh, faces_mesh, origin_marker_mesh)
    """
    edge_meshes = []

    # Orientation arrows: trace path 0 → u → u+v → u+v+w
    orientation_arrows = [
        (origin, u),  # 0 → e1
        (origin + u, v),  # e1 → e1 + e2
        (origin + u + v, w),  # e1 + e2 → e1 + e2 + e3
    ]

    # Edges that are arrows (as endpoint pairs for comparison)
    arrow_edges = {
        (tuple(origin.round(10)), tuple((origin + u).round(10))),
        (tuple((origin + u).round(10)), tuple((origin + u + v).round(10))),
        (tuple((origin + u + v).round(10)), tuple((origin + u + v + w).round(10))),
    }

    for start, direction in orientation_arrows:
        arrow = _create_arrow_mesh(start, direction, shaft_radius=shaft_radius)
        if arrow is not None:
            edge_meshes.append(arrow)

    # All 12 cube edges as (start, end) pairs
    cube_edges = [
        # Edges from origin
        (origin, origin + u),
        (origin, origin + v),
        (origin, origin + w),
        # Edges parallel to u
        (origin + v, origin + u + v),
        (origin + w, origin + u + w),
        (origin + v + w, origin + u + v + w),
        # Edges parallel to v
        (origin + u, origin + u + v),
        (origin + w, origin + v + w),
        (origin + u + w, origin + u + v + w),
        # Edges parallel to w
        (origin + u, origin + u + w),
        (origin + v, origin + v + w),
        (origin + u + v, origin + u + v + w),
    ]

    # Make tubes for edges that aren't arrows
    for start, end in cube_edges:
        edge_key = (tuple(start.round(10)), tuple(end.round(10)))
        edge_key_rev = (tuple(end.round(10)), tuple(start.round(10)))
        if edge_key not in arrow_edges and edge_key_rev not in arrow_edges:
            tube = _create_tube_mesh(start, end, radius=shaft_radius)
            if tube is not None:
                edge_meshes.append(tube)

    edges_mesh = edge_meshes[0] if edge_meshes else pv.PolyData()
    for mesh in edge_meshes[1:]:
        edges_mesh = edges_mesh.merge(mesh)

    # 8 corners for faces
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

    # 6 faces
    face_indices = [
        [0, 1, 4, 2],  # bottom (w=0)
        [3, 5, 7, 6],  # top (w=1)
        [0, 1, 5, 3],  # front (v=0)
        [2, 4, 7, 6],  # back (v=1)
        [0, 2, 6, 3],  # left (u=0)
        [1, 4, 7, 5],  # right (u=1)
    ]

    face_meshes = []
    for indices in face_indices:
        quad = pv.Quadrilateral([corners[k] for k in indices])
        face_meshes.append(quad)

    faces_mesh = face_meshes[0]
    for mesh in face_meshes[1:]:
        faces_mesh = faces_mesh.merge(mesh)

    # Origin marker
    marker = _create_origin_marker(origin)

    return edges_mesh, faces_mesh, marker


def create_quadvector_mesh(
    origin: ndarray,
    u: ndarray,
    v: ndarray,
    w: ndarray,
    x: ndarray,
    projection_axes: tuple[int, int, int] = (0, 1, 2),
    shaft_radius: float = 0.006,
    face_opacity: float = 0.12,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """
    Create meshes for a quadvector (4-blade) as a tesseract projection.

    The 4D tesseract is projected to 3D by selecting 3 axes.

    Args:
        origin: Origin point (nD, where n >= 4)
        u, v, w, x: Four spanning vectors (each nD)
        projection_axes: Which 3 axes to project onto (e.g., (0,1,2) for e123)
        shaft_radius: Tube radius for edges
        face_opacity: Opacity for faces

    Returns:
        (edges_mesh, faces_mesh, origin_marker_mesh)
    """
    # A tesseract has 16 vertices (2^4 combinations)
    # Vertices are: origin + any combination of {u, v, w, x}

    # Generate all 16 vertices in nD
    vertices_nd = []
    for bits in range(16):
        vertex = origin.copy()
        if bits & 1:
            vertex = vertex + u
        if bits & 2:
            vertex = vertex + v
        if bits & 4:
            vertex = vertex + w
        if bits & 8:
            vertex = vertex + x
        vertices_nd.append(vertex)

    # Project vertices to 3D
    ax0, ax1, ax2 = projection_axes
    vertices_3d = []
    for v_nd in vertices_nd:
        if len(v_nd) > max(projection_axes):
            v_3d = array([v_nd[ax0], v_nd[ax1], v_nd[ax2]])
        else:
            # Pad with zeros if needed
            v_padded = zeros(max(projection_axes) + 1)
            v_padded[: len(v_nd)] = v_nd
            v_3d = array([v_padded[ax0], v_padded[ax1], v_padded[ax2]])
        vertices_3d.append(v_3d)

    # 32 edges: pairs of vertices differing by exactly one bit
    edge_pairs = []
    for i in range(16):
        for bit in range(4):
            j = i ^ (1 << bit)
            if i < j:  # Avoid duplicates
                edge_pairs.append((i, j))

    # Determine which edges should have arrows (orientation path)
    # Path: 0 -> u -> u+v -> u+v+w -> u+v+w+x
    arrow_edge_indices = {(0, 1), (1, 3), (3, 7), (7, 15)}

    edge_meshes = []
    for i, j in edge_pairs:
        start = vertices_3d[i]
        end = vertices_3d[j]
        direction = end - start

        if (i, j) in arrow_edge_indices or (j, i) in arrow_edge_indices:
            # Create arrow for orientation edges
            arrow = _create_arrow_mesh(start, direction, shaft_radius=shaft_radius)
            if arrow is not None:
                edge_meshes.append(arrow)
        else:
            # Create tube for non-orientation edges
            tube = _create_tube_mesh(start, end, radius=shaft_radius)
            if tube is not None:
                edge_meshes.append(tube)

    edges_mesh = edge_meshes[0] if edge_meshes else pv.PolyData()
    for mesh in edge_meshes[1:]:
        edges_mesh = edges_mesh.merge(mesh)

    # 24 square faces (6 faces per 3D cube, 4 cubes share each face)
    # For a tesseract, faces are defined by pairs of adjacent vectors
    # Each face is determined by fixing 2 vectors and varying the other 2
    face_definitions = []
    for i in range(4):
        for j in range(i + 1, 4):
            # Face in the plane of vectors i and j
            # There are 2^2 = 4 such faces (for each combination of the other 2 vectors)
            other_bits = [k for k in range(4) if k not in (i, j)]
            for ob0 in [0, 1]:
                for ob1 in [0, 1]:
                    base_bits = (ob0 << other_bits[0]) | (ob1 << other_bits[1])
                    # Four corners of this face
                    corners = [
                        base_bits,
                        base_bits | (1 << i),
                        base_bits | (1 << i) | (1 << j),
                        base_bits | (1 << j),
                    ]
                    face_definitions.append(corners)

    face_meshes = []
    for corners in face_definitions:
        quad_corners = [vertices_3d[c] for c in corners]
        quad = pv.Quadrilateral(quad_corners)
        face_meshes.append(quad)

    faces_mesh = face_meshes[0] if face_meshes else pv.PolyData()
    for mesh in face_meshes[1:]:
        faces_mesh = faces_mesh.merge(mesh)

    # Origin marker (at projected origin)
    origin_3d = vertices_3d[0]
    marker = _create_origin_marker(origin_3d)

    return edges_mesh, faces_mesh, marker


def create_frame_mesh(
    origin: ndarray,
    vectors: ndarray,
    shaft_radius: float = 0.008,
    projection_axes: tuple[int, int, int] | None = None,
    filled: bool = False,
) -> tuple[pv.PolyData | None, pv.PolyData | None, pv.PolyData]:
    """
    Create meshes for a frame (k arrows from origin).

    A frame is visualized as k arrows from origin, one for each spanning vector.
    When filled=True, also draws the edges and faces of the parallelepiped/
    parallelogram they span.

    Args:
        origin: Origin point (nD)
        vectors: Frame vectors, shape (k, d)
        shaft_radius: Arrow shaft radius
        projection_axes: For dim >= 4, which 3 axes to project onto
        filled: If True, draw edges and faces of spanned shape

    Returns:
        (edges_mesh, faces_mesh, origin_marker_mesh)
        faces_mesh is None if filled=False
    """
    origin = array(origin, dtype=float)
    vectors = array(vectors, dtype=float)

    # Ensure vectors is 2D
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    k = vectors.shape[0]

    # Project to 3D if needed
    def project_to_3d(vec, axes):
        if axes is None or len(vec) <= 3:
            v = vec[:3] if len(vec) >= 3 else array([*vec, *[0.0] * (3 - len(vec))])
            return v
        return array([vec[axes[0]], vec[axes[1]], vec[axes[2]]])

    origin_3d = project_to_3d(origin, projection_axes)
    vecs_3d = [project_to_3d(vectors[i], projection_axes) for i in range(k)]

    edge_meshes = []

    # Create k arrows from origin
    for i in range(k):
        arrow = _create_arrow_mesh(origin_3d, vecs_3d[i], shaft_radius=shaft_radius)
        if arrow is not None:
            edge_meshes.append(arrow)

    faces_mesh = None

    if filled and k >= 2:
        # Add the non-origin edges and faces based on k
        if k == 2:
            # Parallelogram: 2 more edges (completing the quad)
            u, v = vecs_3d[0], vecs_3d[1]
            # Edges: u → u+v and v → u+v
            tube1 = _create_tube_mesh(origin_3d + u, origin_3d + u + v, radius=shaft_radius * 0.75)
            tube2 = _create_tube_mesh(origin_3d + v, origin_3d + u + v, radius=shaft_radius * 0.75)
            if tube1 is not None:
                edge_meshes.append(tube1)
            if tube2 is not None:
                edge_meshes.append(tube2)
            # Face
            corners = [origin_3d, origin_3d + u, origin_3d + u + v, origin_3d + v]
            faces_mesh = pv.Quadrilateral(corners)

        elif k == 3:
            # Parallelepiped: 9 more edges + 6 faces
            u, v, w = vecs_3d[0], vecs_3d[1], vecs_3d[2]

            # All 12 edges, but we already have the 3 from origin as arrows
            # Remaining 9 edges as tubes
            remaining_edges = [
                # Edges parallel to u (not from origin)
                (origin_3d + v, origin_3d + u + v),
                (origin_3d + w, origin_3d + u + w),
                (origin_3d + v + w, origin_3d + u + v + w),
                # Edges parallel to v (not from origin)
                (origin_3d + u, origin_3d + u + v),
                (origin_3d + w, origin_3d + v + w),
                (origin_3d + u + w, origin_3d + u + v + w),
                # Edges parallel to w (not from origin)
                (origin_3d + u, origin_3d + u + w),
                (origin_3d + v, origin_3d + v + w),
                (origin_3d + u + v, origin_3d + u + v + w),
            ]
            for start, end in remaining_edges:
                tube = _create_tube_mesh(start, end, radius=shaft_radius * 0.75)
                if tube is not None:
                    edge_meshes.append(tube)

            # 6 faces of the parallelepiped
            face_quads = [
                # Bottom (z=0 plane)
                [origin_3d, origin_3d + u, origin_3d + u + v, origin_3d + v],
                # Top (z=w plane)
                [origin_3d + w, origin_3d + u + w, origin_3d + u + v + w, origin_3d + v + w],
                # Front (y=0 plane)
                [origin_3d, origin_3d + u, origin_3d + u + w, origin_3d + w],
                # Back (y=v plane)
                [origin_3d + v, origin_3d + u + v, origin_3d + u + v + w, origin_3d + v + w],
                # Left (x=0 plane)
                [origin_3d, origin_3d + v, origin_3d + v + w, origin_3d + w],
                # Right (x=u plane)
                [origin_3d + u, origin_3d + u + v, origin_3d + u + v + w, origin_3d + u + w],
            ]
            face_meshes = [pv.Quadrilateral(corners) for corners in face_quads]
            faces_mesh = face_meshes[0]
            for fm in face_meshes[1:]:
                faces_mesh = faces_mesh.merge(fm)

        elif k == 4:
            # Tesseract projection: more complex, use quadvector pattern
            u, v, w, x = vecs_3d[0], vecs_3d[1], vecs_3d[2], vecs_3d[3]

            # 32 edges total, 4 from origin as arrows, 28 as tubes
            # Generate all vertices of the tesseract
            vertices = []
            for i in range(16):
                vertex = origin_3d.copy()
                if i & 1:
                    vertex = vertex + u
                if i & 2:
                    vertex = vertex + v
                if i & 4:
                    vertex = vertex + w
                if i & 8:
                    vertex = vertex + x
                vertices.append(vertex)

            # All 32 edges (vertices differing by exactly one bit)
            all_edges = []
            for i in range(16):
                for bit in range(4):
                    j = i ^ (1 << bit)
                    if i < j:
                        all_edges.append((vertices[i], vertices[j]))

            # Origin edges (from vertex 0) are already arrows
            for start, end in all_edges:
                # Skip edges from origin (vertex 0)
                is_origin_edge = norm(start - origin_3d) < 1e-10 and any(
                    norm(end - origin_3d - vecs_3d[b]) < 1e-10 for b in range(4)
                )
                if not is_origin_edge:
                    tube = _create_tube_mesh(start, end, radius=shaft_radius * 0.75)
                    if tube is not None:
                        edge_meshes.append(tube)

            # 24 faces (each face is a 2D face of the tesseract)
            # Each face is determined by fixing 2 coordinates
            face_quads = []
            for fixed1 in range(4):
                for fixed2 in range(fixed1 + 1, 4):
                    for val1 in [0, 1]:
                        for val2 in [0, 1]:
                            # Get corners where fixed1=val1 and fixed2=val2
                            corners = []
                            for bits in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                                idx = 0
                                bit_pos = 0
                                for dim in range(4):
                                    if dim == fixed1:
                                        if val1:
                                            idx |= 1 << dim
                                    elif dim == fixed2:
                                        if val2:
                                            idx |= 1 << dim
                                    else:
                                        if bits[bit_pos]:
                                            idx |= 1 << dim
                                        bit_pos += 1
                                corners.append(vertices[idx])
                            face_quads.append(corners)

            face_meshes = [pv.Quadrilateral(corners) for corners in face_quads]
            faces_mesh = face_meshes[0]
            for fm in face_meshes[1:]:
                faces_mesh = faces_mesh.merge(fm)

    # Combine all edge meshes
    if edge_meshes:
        edges_mesh = edge_meshes[0]
        for mesh in edge_meshes[1:]:
            edges_mesh = edges_mesh.merge(mesh)
    else:
        edges_mesh = None

    # Create origin marker
    origin_mesh = _create_origin_marker(origin_3d)

    return edges_mesh, faces_mesh, origin_mesh


def create_blade_mesh(
    grade: int,
    origin: ndarray,
    vectors: ndarray,
    shaft_radius: float = 0.008,
    edge_radius: float = 0.006,
    projection_axes: tuple[int, int, int] | None = None,
) -> tuple[pv.PolyData | None, pv.PolyData | None, pv.PolyData]:
    """
    Create meshes for a blade of any grade.

    This is the main entry point for mesh creation.

    Args:
        grade: Blade grade (1, 2, 3, or 4)
        origin: Origin point (nD)
        vectors: Spanning vectors (shape depends on grade)
        shaft_radius: Arrow shaft radius (for vectors/bivectors)
        edge_radius: Edge tube radius (for trivectors/quadvectors)
        projection_axes: For grade >= 4, which 3 axes to project onto

    Returns:
        (edges_mesh, faces_mesh, origin_marker_mesh)
        - For grade 1: edges_mesh is the arrow, faces_mesh is None
        - For grade 2: edges_mesh is circulation arrows, faces_mesh is the quad
        - For grade 3: edges_mesh is tubes, faces_mesh is 6 quads
        - For grade 4: edges_mesh is tubes (tesseract), faces_mesh is 24 quads
    """
    origin = array(origin, dtype=float)
    vectors = array(vectors, dtype=float)

    if grade == 1:
        direction = vectors[0] if vectors.ndim > 1 else vectors
        # Project to 3D if needed
        if len(direction) > 3:
            axes = projection_axes or (0, 1, 2)
            direction = array([direction[axes[0]], direction[axes[1]], direction[axes[2]]])
            origin_3d = array([origin[axes[0]], origin[axes[1]], origin[axes[2]]])
        else:
            origin_3d = origin[:3] if len(origin) >= 3 else array([*origin, *[0.0] * (3 - len(origin))])
            direction = direction[:3] if len(direction) >= 3 else array([*direction, *[0.0] * (3 - len(direction))])
        arrow, marker = create_vector_mesh(origin_3d, direction, shaft_radius=shaft_radius)
        return arrow, None, marker

    elif grade == 2:
        u, v = vectors[0], vectors[1]
        # Project to 3D if needed
        if len(u) > 3:
            axes = projection_axes or (0, 1, 2)
            u = array([u[axes[0]], u[axes[1]], u[axes[2]]])
            v = array([v[axes[0]], v[axes[1]], v[axes[2]]])
            origin_3d = array([origin[axes[0]], origin[axes[1]], origin[axes[2]]])
        else:
            origin_3d = origin[:3] if len(origin) >= 3 else array([*origin, *[0.0] * (3 - len(origin))])
            u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
            v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
        return create_bivector_mesh(origin_3d, u, v, shaft_radius=edge_radius)

    elif grade == 3:
        u, v, w = vectors[0], vectors[1], vectors[2]
        # Project to 3D if needed
        if len(u) > 3:
            axes = projection_axes or (0, 1, 2)
            u = array([u[axes[0]], u[axes[1]], u[axes[2]]])
            v = array([v[axes[0]], v[axes[1]], v[axes[2]]])
            w = array([w[axes[0]], w[axes[1]], w[axes[2]]])
            origin_3d = array([origin[axes[0]], origin[axes[1]], origin[axes[2]]])
        else:
            origin_3d = origin[:3] if len(origin) >= 3 else array([*origin, *[0.0] * (3 - len(origin))])
            u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
            v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
            w = w[:3] if len(w) >= 3 else array([*w, *[0.0] * (3 - len(w))])
        return create_trivector_mesh(origin_3d, u, v, w, shaft_radius=edge_radius)

    elif grade == 4:
        u, v, w, x = vectors[0], vectors[1], vectors[2], vectors[3]
        axes = projection_axes or (0, 1, 2)
        return create_quadvector_mesh(origin, u, v, w, x, projection_axes=axes, shaft_radius=edge_radius)

    else:
        raise NotImplementedError(f"Grade {grade} not supported")


# =============================================================================
# Drawing Primitives (add meshes to plotter)
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
    mesh = _create_arrow_mesh(start, direction, shaft_radius, tip_ratio, tip_radius_ratio, resolution)
    if mesh is not None:
        plotter.add_mesh(mesh, color=color, smooth_shading=True)


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
    """Draw a parallelogram with oriented boundary (circulation arrows)."""
    edges_mesh, face_mesh, marker = create_bivector_mesh(origin, u, v, shaft_radius=edge_radius)

    if tetrad:
        plotter.add_mesh(edges_mesh, color=color, smooth_shading=True)
        plotter.add_mesh(marker, color=color, smooth_shading=True)

    if surface:
        plotter.add_mesh(face_mesh, color=color, opacity=opacity, smooth_shading=True)


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
    """Draw a parallelepiped with origin marker."""
    edges_mesh, faces_mesh, marker = create_trivector_mesh(origin, u, v, w, edge_radius=edge_radius)

    if tetrad:
        plotter.add_mesh(edges_mesh, color=color, smooth_shading=True)
        plotter.add_mesh(marker, color=color, smooth_shading=True)

    if surface:
        plotter.add_mesh(faces_mesh, color=color, opacity=opacity, smooth_shading=True)


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
        vectors = _factor_blade_to_arrays(b)
        u, v = vectors[0], vectors[1]
        u = u[:3] if len(u) >= 3 else array([*u, *[0.0] * (3 - len(u))])
        v = v[:3] if len(v) >= 3 else array([*v, *[0.0] * (3 - len(v))])
        _draw_parallelogram(plotter, origin, u, v, color, tetrad=tetrad, surface=surface, edge_radius=edge_radius)

    elif grade == 3:
        # Trivector: factor and draw parallelepiped
        vectors = _factor_blade_to_arrays(b)
        u, v, w = vectors[0], vectors[1], vectors[2]
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
            vectors = _factor_blade_to_arrays(b)
            u, v = vectors[0], vectors[1]
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
            vectors = _factor_blade_to_arrays(b)
            u, v, w = vectors[0], vectors[1], vectors[2]
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
    labels: tuple[str, str, str] | None = None,
) -> list | None:
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
        label: If True, add labels
        label_offset: Distance to offset labels from axes
        labels: Custom labels for (x, y, z) axes. Default: e1, e2, e3

    Returns:
        List of label actors if label=True, else None. Can be used to
        remove/update labels later.
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

    if labels is not None:
        names = list(labels)
    else:
        names = [r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_3$"]

    label_actors = []

    for direction, offset_dir, axis_name in zip(directions, offset_dirs, names, strict=False):
        # Draw the arrow
        if tetrad:
            _draw_arrow(plotter, zeros(3), direction, color, shaft_radius=0.004, tip_ratio=0.08)

        # Add label
        if label:
            offset = offset_dir / norm(offset_dir) * label_offset
            label_pos = direction * 0.5 + offset
            actor = plotter.add_point_labels(
                [label_pos],
                [axis_name],
                font_size=12,
                text_color=color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )
            label_actors.append(actor)

    return label_actors if label else None


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

    # Map math basis indices (e₁, e₂, e₃) to direction vectors
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
