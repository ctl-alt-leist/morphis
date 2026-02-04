"""
PyVista Backend Implementation

Implements the RenderBackend protocol using PyVista/VTK for 3D rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from numpy import array, zeros
from numpy.linalg import norm
from numpy.typing import NDArray

from morphis.visuals.theme import Color, Theme


if TYPE_CHECKING:
    from pyvista import Plotter, PolyData


class TrackedObject:
    """Internal tracking for rendered objects."""

    def __init__(
        self,
        object_id: str,
        object_type: str,
        actors: list[Any],
        color: Color,
        opacity: float,
        extra: dict | None = None,
    ):
        self.object_id = object_id
        self.object_type = object_type
        self.actors = actors  # List of VTK actors
        self.color = color
        self.opacity = opacity
        self.extra = extra or {}


class PyVistaBackend:
    """
    PyVista/VTK rendering backend.

    Implements the RenderBackend protocol for 3D visualization.
    """

    def __init__(self):
        self._plotter: Plotter | None = None
        self._objects: dict[str, TrackedObject] = {}
        self._theme: Theme | None = None
        self._size: tuple[int, int] = (1200, 900)
        self._show_basis: bool = True
        self._basis_actors: list[Any] = []
        self._current_basis_labels: tuple[str, str, str] | None = None
        self._explicit_clipping_range: tuple[float, float] | None = None
        self._lights: dict[str, Any] = {}  # light_id -> pyvista.Light

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(
        self,
        size: tuple[int, int],
        theme: Theme,
        show_basis: bool = True,
    ) -> None:
        """Initialize the rendering window."""
        from pyvista import Plotter

        self._theme = theme
        self._size = size
        self._show_basis = show_basis

        self._plotter = Plotter(off_screen=False)
        self._plotter.set_background(theme.background)
        self._plotter.window_size = size

        if show_basis:
            self._draw_coordinate_basis()

    def close(self) -> None:
        """Close the rendering window."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        self._objects.clear()
        self._basis_actors.clear()

    def _ensure_plotter(self):
        """Ensure plotter is initialized."""
        if self._plotter is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

    # =========================================================================
    # Mesh Operations
    # =========================================================================

    def add_mesh(
        self,
        vertices: NDArray,
        faces: NDArray,
        color: Color,
        opacity: float = 1.0,
        smooth_shading: bool = True,
        show_edges: bool = False,
    ) -> str:
        """Add a mesh to the scene."""
        from pyvista import PolyData

        self._ensure_plotter()
        object_id = str(uuid4())

        mesh = PolyData(vertices, faces)
        actor = self._plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            smooth_shading=smooth_shading,
            show_edges=show_edges,
        )

        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            object_type="mesh",
            actors=[actor],
            color=color,
            opacity=opacity,
            extra={"mesh": mesh},
        )

        return object_id

    def update_mesh(self, object_id: str, vertices: NDArray) -> None:
        """Update mesh vertex positions."""
        if object_id not in self._objects:
            return

        tracked = self._objects[object_id]
        if tracked.extra.get("mesh"):
            tracked.extra["mesh"].points = vertices

    # =========================================================================
    # Arrow Operations
    # =========================================================================

    def add_arrows(
        self,
        origins: NDArray,
        directions: NDArray,
        color: Color,
        opacity: float = 1.0,
        tip_length: float = 0.12,
        tip_radius: float = 0.02,
        shaft_radius: float = 0.008,
    ) -> str:
        """Add arrows to the scene."""
        self._ensure_plotter()
        object_id = str(uuid4())

        origins = array(origins, dtype=float)
        directions = array(directions, dtype=float)

        # Handle single arrow (1D arrays)
        if origins.ndim == 1:
            origins = origins.reshape(1, -1)
            directions = directions.reshape(1, -1)

        actors = []
        arrow_meshes = []

        for origin, direction in zip(origins, directions, strict=False):
            arrow_mesh = self._create_arrow_mesh(origin, direction, shaft_radius, tip_length, tip_radius)
            if arrow_mesh is not None:
                actor = self._plotter.add_mesh(arrow_mesh, color=color, opacity=opacity, smooth_shading=True)
                actors.append(actor)
                arrow_meshes.append(arrow_mesh)

        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            object_type="arrows",
            actors=actors,
            color=color,
            opacity=opacity,
            extra={"origins": origins.copy(), "directions": directions.copy(), "meshes": arrow_meshes},
        )

        return object_id

    def update_arrows(
        self,
        object_id: str,
        origins: NDArray,
        directions: NDArray,
    ) -> None:
        """Update arrow positions and directions."""
        if object_id not in self._objects:
            return

        tracked = self._objects[object_id]
        origins = array(origins, dtype=float)
        directions = array(directions, dtype=float)

        if origins.ndim == 1:
            origins = origins.reshape(1, -1)
            directions = directions.reshape(1, -1)

        # Update each arrow mesh
        meshes = tracked.extra.get("meshes", [])
        for idx, (origin, direction, _mesh) in enumerate(zip(origins, directions, meshes, strict=False)):
            new_mesh = self._create_arrow_mesh(origin, direction)
            if new_mesh is not None and tracked.actors[idx] is not None:
                tracked.actors[idx].mapper.SetInputData(new_mesh)

        tracked.extra["origins"] = origins.copy()
        tracked.extra["directions"] = directions.copy()

    def _create_arrow_mesh(
        self,
        start: NDArray,
        direction: NDArray,
        shaft_radius: float = 0.008,
        tip_ratio: float = 0.12,
        tip_radius_ratio: float = 2.5,
    ) -> "PolyData | None":
        """Create an arrow mesh."""
        from pyvista import Cone, Cylinder

        length = norm(direction)
        if length < 1e-10:
            return None

        dir_norm = direction / length
        tip_length = length * tip_ratio
        shaft_length = length - tip_length
        tip_radius = shaft_radius * tip_radius_ratio

        shaft_end = start + dir_norm * shaft_length

        shaft = Cylinder(
            center=(start + shaft_end) / 2,
            direction=dir_norm,
            radius=shaft_radius,
            height=shaft_length,
            resolution=20,
            capping=True,
        )

        tip = Cone(
            center=shaft_end + dir_norm * (tip_length / 2),
            direction=dir_norm,
            height=tip_length,
            radius=tip_radius,
            resolution=20,
            capping=True,
        )

        return shaft.merge(tip)

    # =========================================================================
    # Point Operations
    # =========================================================================

    def add_points(
        self,
        positions: NDArray,
        color: Color,
        opacity: float = 1.0,
        point_size: float = 5.0,
    ) -> str:
        """Add points to the scene."""
        from pyvista import PolyData

        self._ensure_plotter()
        object_id = str(uuid4())

        positions = array(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        points = PolyData(positions)
        actor = self._plotter.add_mesh(
            points,
            color=color,
            opacity=opacity,
            point_size=point_size,
            render_points_as_spheres=True,
        )

        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            object_type="points",
            actors=[actor],
            color=color,
            opacity=opacity,
            extra={"mesh": points},
        )

        return object_id

    def update_points(self, object_id: str, positions: NDArray) -> None:
        """Update point positions."""
        if object_id not in self._objects:
            return

        tracked = self._objects[object_id]
        positions = array(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        if tracked.extra.get("mesh"):
            tracked.extra["mesh"].points = positions

    # =========================================================================
    # Line Operations
    # =========================================================================

    def add_lines(
        self,
        points: NDArray,
        color: Color,
        opacity: float = 1.0,
        line_width: float = 2.0,
    ) -> str:
        """Add a polyline to the scene."""
        from pyvista import PolyData

        self._ensure_plotter()
        object_id = str(uuid4())

        points = array(points, dtype=float)
        n_points = len(points)

        # Create line connectivity
        cells = array([n_points] + list(range(n_points)))

        mesh = PolyData(points, lines=cells)
        actor = self._plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            line_width=line_width,
        )

        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            object_type="lines",
            actors=[actor],
            color=color,
            opacity=opacity,
            extra={"mesh": mesh},
        )

        return object_id

    def update_lines(self, object_id: str, points: NDArray) -> None:
        """Update line vertices."""
        if object_id not in self._objects:
            return

        tracked = self._objects[object_id]
        points = array(points, dtype=float)

        if tracked.extra.get("mesh"):
            tracked.extra["mesh"].points = points

    # =========================================================================
    # Span (Bivector) Operations
    # =========================================================================

    def add_span(
        self,
        origin: NDArray,
        vectors: NDArray,
        color: Color,
        opacity: float = 0.3,
        filled: bool = True,
    ) -> str:
        """Add a span (parallelogram/parallelepiped) to the scene."""
        from pyvista import Quadrilateral

        self._ensure_plotter()
        object_id = str(uuid4())

        origin = array(origin, dtype=float)
        vectors = array(vectors, dtype=float)

        actors = []

        if len(vectors) == 2:
            # Parallelogram (bivector)
            u, v = vectors[0], vectors[1]
            corners = [origin, origin + u, origin + u + v, origin + v]

            if filled:
                face = Quadrilateral(corners)
                actor = self._plotter.add_mesh(face, color=color, opacity=opacity, smooth_shading=True)
                actors.append(actor)

            # Add edge outline
            edge_points = array([*corners, corners[0]])
            self._add_curve_internal(edge_points, color, 1.0, 0.003, actors)

        elif len(vectors) == 3:
            # Parallelepiped (trivector)
            u, v, w = vectors[0], vectors[1], vectors[2]

            # 6 faces
            face_quads = [
                [origin, origin + u, origin + u + v, origin + v],
                [origin + w, origin + u + w, origin + u + v + w, origin + v + w],
                [origin, origin + u, origin + u + w, origin + w],
                [origin + v, origin + u + v, origin + u + v + w, origin + v + w],
                [origin, origin + v, origin + v + w, origin + w],
                [origin + u, origin + u + v, origin + u + v + w, origin + u + w],
            ]

            if filled:
                for corners in face_quads:
                    face = Quadrilateral(corners)
                    actor = self._plotter.add_mesh(face, color=color, opacity=opacity, smooth_shading=True)
                    actors.append(actor)

        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            object_type="span",
            actors=actors,
            color=color,
            opacity=opacity,
            extra={"origin": origin.copy(), "vectors": vectors.copy()},
        )

        return object_id

    def update_span(
        self,
        object_id: str,
        origin: NDArray,
        vectors: NDArray,
    ) -> None:
        """Update span geometry."""
        # For now, remove and re-add (spans are complex to update in-place)
        if object_id in self._objects:
            tracked = self._objects[object_id]
            self.remove(object_id)
            new_id = self.add_span(origin, vectors, tracked.color, tracked.opacity, filled=True)
            # Keep the old ID for consistency
            self._objects[object_id] = self._objects.pop(new_id)

    def _add_curve_internal(
        self,
        points: NDArray,
        color: Color,
        opacity: float,
        radius: float,
        actors: list,
    ) -> None:
        """Internal helper to add tube along points."""
        from pyvista import Cylinder

        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            direction = end - start
            length = norm(direction)

            if length < 1e-10:
                continue

            tube = Cylinder(
                center=(start + end) / 2,
                direction=direction / length,
                radius=radius,
                height=length,
                resolution=12,
                capping=True,
            )
            actor = self._plotter.add_mesh(tube, color=color, opacity=opacity, smooth_shading=True)
            actors.append(actor)

    # =========================================================================
    # Text Operations
    # =========================================================================

    def add_text(
        self,
        text: str,
        position: NDArray,
        color: Color,
        font_size: int = 12,
        anchor: str = "center",
    ) -> str:
        """Add 3D text annotation to the scene."""
        self._ensure_plotter()
        object_id = str(uuid4())

        position = array(position, dtype=float)

        actor = self._plotter.add_point_labels(
            [position],
            [text],
            font_size=font_size,
            text_color=color,
            point_size=0,
            shape=None,
            show_points=False,
            always_visible=True,
        )

        self._objects[object_id] = TrackedObject(
            object_id=object_id,
            object_type="text",
            actors=[actor],
            color=color,
            opacity=1.0,
            extra={"text": text, "position": position.copy()},
        )

        return object_id

    def update_text(
        self,
        object_id: str,
        text: str | None = None,
        position: NDArray | None = None,
    ) -> None:
        """Update text content and/or position."""
        if object_id not in self._objects:
            return

        tracked = self._objects[object_id]

        # PyVista text labels are not easily updatable - remove and re-add
        new_text = text if text is not None else tracked.extra.get("text", "")
        new_pos = array(position, dtype=float) if position is not None else tracked.extra.get("position", zeros(3))

        # Remove old
        for actor in tracked.actors:
            self._plotter.remove_actor(actor)

        # Add new
        actor = self._plotter.add_point_labels(
            [new_pos],
            [new_text],
            font_size=12,
            text_color=tracked.color,
            point_size=0,
            shape=None,
            show_points=False,
            always_visible=True,
        )

        tracked.actors = [actor]
        tracked.extra["text"] = new_text
        tracked.extra["position"] = new_pos

    # =========================================================================
    # Object Management
    # =========================================================================

    def set_opacity(self, object_id: str, opacity: float) -> None:
        """Set object opacity."""
        if object_id not in self._objects:
            return

        tracked = self._objects[object_id]
        tracked.opacity = opacity

        for actor in tracked.actors:
            if actor is not None:
                actor.GetProperty().SetOpacity(opacity)

    def remove(self, object_id: str) -> None:
        """Remove an object from the scene."""
        if object_id not in self._objects:
            return

        tracked = self._objects.pop(object_id)

        for actor in tracked.actors:
            if actor is not None:
                self._plotter.remove_actor(actor)

    # =========================================================================
    # Camera
    # =========================================================================

    def set_camera(
        self,
        position: tuple[float, float, float] | None = None,
        focal_point: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
    ) -> None:
        """Set camera position and orientation."""
        self._ensure_plotter()

        if position is not None:
            self._plotter.camera.position = position
        if focal_point is not None:
            self._plotter.camera.focal_point = focal_point
        if up is not None:
            self._plotter.camera.up = up

    def reset_camera(self) -> None:
        """Reset camera to fit all objects in view."""
        self._ensure_plotter()
        self._plotter.reset_camera()

    def reset_clipping_range(self) -> None:
        """Reset camera clipping range to fit current scene."""
        self._ensure_plotter()
        self._plotter.reset_camera_clipping_range()

    def set_clipping_range(self, near: float, far: float) -> None:
        """
        Set camera clipping range explicitly.

        Args:
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self._ensure_plotter()
        self._explicit_clipping_range = (near, far)
        self._plotter.camera.clipping_range = (near, far)

    def _ensure_clipping_range(self) -> None:
        """Reapply explicit clipping range if set (PyVista may auto-reset it)."""
        if self._explicit_clipping_range is not None:
            self._plotter.camera.clipping_range = self._explicit_clipping_range

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(self) -> None:
        """Render the current frame."""
        if self._plotter is not None:
            self._ensure_clipping_range()
            self._plotter.render()

    def capture_frame(self) -> NDArray:
        """Capture current frame as image."""
        self._ensure_plotter()
        return self._plotter.screenshot(return_img=True)

    def show(self, interactive: bool = True) -> None:
        """Display the scene."""
        self._ensure_plotter()
        self._ensure_clipping_range()
        self._plotter.show(interactive_update=True, auto_close=False)

    # =========================================================================
    # Basis Display
    # =========================================================================

    def _draw_coordinate_basis(self) -> None:
        """Draw coordinate basis arrows."""
        if self._theme is None:
            return

        color = self._theme.axis_color
        directions = [
            array([1.0, 0.0, 0.0]),
            array([0.0, 1.0, 0.0]),
            array([0.0, 0.0, 1.0]),
        ]
        origin = zeros(3)

        for direction in directions:
            arrow = self._create_arrow_mesh(origin, direction, shaft_radius=0.004, tip_ratio=0.08)
            if arrow is not None:
                actor = self._plotter.add_mesh(arrow, color=color, smooth_shading=True)
                self._basis_actors.append(actor)

        # Add labels
        labels = self._current_basis_labels or (r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_3$")
        self._add_basis_labels(labels)

    def _add_basis_labels(self, labels: tuple[str, str, str]) -> None:
        """Add labels to basis arrows."""
        directions = [
            array([1.0, 0.0, 0.0]),
            array([0.0, 1.0, 0.0]),
            array([0.0, 0.0, 1.0]),
        ]
        offset_dirs = [
            array([0, -1, -1]),
            array([-1, 0, -1]),
            array([-1, -1, 0]),
        ]
        label_offset = 0.08
        color = self._theme.axis_color if self._theme else (0.7, 0.7, 0.7)

        for direction, offset_dir, label in zip(directions, offset_dirs, labels, strict=False):
            offset = offset_dir / norm(offset_dir) * label_offset
            label_pos = direction * 0.5 + offset
            actor = self._plotter.add_point_labels(
                [label_pos],
                [label],
                font_size=12,
                text_color=color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )
            self._basis_actors.append(actor)

    def set_basis_labels(self, labels: tuple[str, str, str]) -> None:
        """Update coordinate basis labels."""
        if labels == self._current_basis_labels:
            return

        self._current_basis_labels = labels

        if self._plotter is None or not self._show_basis:
            return

        # Remove old label actors (keep arrow actors which are first 3)
        for actor in self._basis_actors[3:]:
            self._plotter.remove_actor(actor)
        self._basis_actors = self._basis_actors[:3]

        # Add new labels
        self._add_basis_labels(labels)

    # =========================================================================
    # Properties
    # =========================================================================

    # =========================================================================
    # Lighting
    # =========================================================================

    def add_light(
        self,
        position: tuple[float, float, float] = (1, 1, 1),
        focal_point: tuple[float, float, float] = (0, 0, 0),
        intensity: float = 1.0,
        color: Color = (1.0, 1.0, 1.0),
        directional: bool = True,
        attenuation: tuple[float, float, float] | None = None,
    ) -> str:
        """
        Add a light to the scene.

        Args:
            position: Light position (or direction for directional lights)
            focal_point: Point the light aims at
            intensity: Light brightness
            color: Light color RGB
            directional: If True, parallel rays (sun-like). If False, point light.
            attenuation: For positional lights: (constant, linear, quadratic).
                        None = no falloff.

        Returns:
            Light ID
        """
        from pyvista import Light

        self._ensure_plotter()
        light_id = str(uuid4())

        light = Light(
            position=position,
            focal_point=focal_point,
            intensity=intensity,
            color=color,
        )

        # Directional = parallel rays from infinity (like sun)
        # Positional = point light with actual position
        light.positional = not directional

        if not directional and attenuation is not None:
            # Set attenuation for positional lights
            # VTK uses: intensity / (c + l*d + q*d²)
            light.attenuation_values = attenuation

        self._plotter.add_light(light)
        self._lights[light_id] = light

        return light_id

    def remove_light(self, light_id: str) -> None:
        """Remove a light from the scene."""
        if light_id not in self._lights:
            return

        light = self._lights.pop(light_id)
        if self._plotter is not None:
            self._plotter.remove_actor(light)

    def clear_lights(self) -> None:
        """Remove all user-added lights."""
        for light_id in list(self._lights.keys()):
            self.remove_light(light_id)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def plotter(self) -> "Plotter | None":
        """Access underlying PyVista plotter (for advanced use)."""
        return self._plotter
