"""
Renderer - Low-level visualization layer

The Renderer manages VTK/PyVista actors and tracks objects that can be
dynamically updated. It delegates all mesh creation to drawing.py.

It has no knowledge of:
- Geometric algebra (blades, motors, etc.)
- Time or animation
- Effects or scheduling
"""

import pyvista as pv
from dataclasses import dataclass
from numpy import array
from numpy.typing import NDArray

from morphis.visualization.drawing import create_blade_mesh
from morphis.visualization.theme import Color, Theme, get_theme


@dataclass
class TrackedObject:
    """Internal state for a rendered object."""

    obj_id: int
    grade: int
    color: Color
    edges_actor: pv.Actor
    faces_actor: pv.Actor | None
    origin_actor: pv.Actor | None
    opacity: float = 1.0


class Renderer:
    """
    Low-level renderer for 3D geometric objects.

    Manages a PyVista plotter and tracks objects that can be dynamically
    updated. Each object is identified by a unique ID.

    Example:
        renderer = Renderer(theme="obsidian")
        renderer.add_object(
            obj_id=1,
            grade=3,
            origin=array([0, 0, 0]),
            vectors=array([[1,0,0], [0,1,0], [0,0,1]]),
            color=(0.85, 0.2, 0.2),
        )
        renderer.show()
        renderer.update_object(1, new_vectors, opacity=0.5)
        renderer.render()
    """

    def __init__(
        self,
        theme: str | Theme = "obsidian",
        size: tuple[int, int] = (1800, 1350),
        show_basis: bool = True,
    ):
        if isinstance(theme, str):
            theme = get_theme(theme)

        self.theme = theme
        self._size = size
        self._show_basis = show_basis
        self._plotter: pv.Plotter | None = None
        self._objects: dict[int, TrackedObject] = {}
        self._color_index = 0

    def _ensure_plotter(self):
        """Create plotter if not yet created."""
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=False)
            self._plotter.set_background(self.theme.background)
            self._plotter.window_size = self._size

            if self._show_basis:
                from morphis.visualization.drawing import draw_coordinate_basis

                draw_coordinate_basis(self._plotter, color=self.theme.axis_color)

    def _next_color(self) -> Color:
        """Get the next color from the theme palette."""
        color = self.theme.palette[self._color_index % len(self.theme.palette)]
        self._color_index += 1
        return color

    # =========================================================================
    # Object Management
    # =========================================================================

    def add_object(
        self,
        obj_id: int,
        grade: int,
        origin: NDArray,
        vectors: NDArray,
        color: Color | None = None,
        opacity: float = 1.0,
    ):
        """
        Add a new object to the renderer.

        Args:
            obj_id: Unique identifier for this object
            grade: Geometric grade (1=vector, 2=bivector, 3=trivector)
            origin: Origin point (3D)
            vectors: Spanning vectors (shape depends on grade)
            color: RGB color tuple (0-1 range), or None for auto
            opacity: Initial opacity [0, 1]
        """
        self._ensure_plotter()

        if obj_id in self._objects:
            self.update_object(obj_id, origin, vectors, opacity)
            return

        if color is None:
            color = self._next_color()

        origin = array(origin, dtype=float)
        vectors = array(vectors, dtype=float)

        # Create meshes using centralized drawing functions
        edges_mesh, faces_mesh, origin_mesh = create_blade_mesh(grade, origin, vectors)

        # Add to plotter
        edges_actor = None
        if edges_mesh is not None:
            edges_actor = self._plotter.add_mesh(
                edges_mesh, color=color, opacity=opacity, smooth_shading=True
            )

        faces_actor = None
        if faces_mesh is not None:
            face_opacity = opacity * (0.25 if grade == 2 else 0.2)
            faces_actor = self._plotter.add_mesh(
                faces_mesh, color=color, opacity=face_opacity, smooth_shading=True
            )

        origin_actor = self._plotter.add_mesh(
            origin_mesh, color=color, opacity=opacity, smooth_shading=True
        )

        self._objects[obj_id] = TrackedObject(
            obj_id=obj_id,
            grade=grade,
            color=color,
            edges_actor=edges_actor,
            faces_actor=faces_actor,
            origin_actor=origin_actor,
            opacity=opacity,
        )

    def update_object(
        self,
        obj_id: int,
        origin: NDArray,
        vectors: NDArray,
        opacity: float | None = None,
    ):
        """
        Update an existing object's geometry and/or opacity.

        Args:
            obj_id: The object to update
            origin: New origin point
            vectors: New spanning vectors
            opacity: New opacity (or None to keep current)
        """
        if obj_id not in self._objects:
            return

        tracked = self._objects[obj_id]
        origin = array(origin, dtype=float)
        vectors = array(vectors, dtype=float)

        if opacity is not None:
            tracked.opacity = opacity

        # Recreate meshes using centralized drawing functions
        edges_mesh, faces_mesh, origin_mesh = create_blade_mesh(
            tracked.grade, origin, vectors
        )

        # Update actors with new meshes
        if tracked.edges_actor is not None and edges_mesh is not None:
            tracked.edges_actor.mapper.SetInputData(edges_mesh)
        if tracked.faces_actor is not None and faces_mesh is not None:
            tracked.faces_actor.mapper.SetInputData(faces_mesh)
        if tracked.origin_actor is not None:
            tracked.origin_actor.mapper.SetInputData(origin_mesh)

        # Update opacity
        if tracked.edges_actor is not None:
            tracked.edges_actor.GetProperty().SetOpacity(tracked.opacity)
        if tracked.faces_actor is not None:
            face_opacity = tracked.opacity * (0.25 if tracked.grade == 2 else 0.2)
            tracked.faces_actor.GetProperty().SetOpacity(face_opacity)
        if tracked.origin_actor is not None:
            tracked.origin_actor.GetProperty().SetOpacity(tracked.opacity)

    def set_opacity(self, obj_id: int, opacity: float):
        """Set the opacity of an object."""
        if obj_id not in self._objects:
            return

        tracked = self._objects[obj_id]
        tracked.opacity = opacity
        tracked.edges_actor.GetProperty().SetOpacity(opacity)
        if tracked.faces_actor is not None:
            face_opacity = opacity * (0.25 if tracked.grade == 2 else 0.2)
            tracked.faces_actor.GetProperty().SetOpacity(face_opacity)
        if tracked.origin_actor is not None:
            tracked.origin_actor.GetProperty().SetOpacity(opacity)

    def remove_object(self, obj_id: int):
        """Remove an object from the renderer."""
        if obj_id not in self._objects:
            return

        tracked = self._objects.pop(obj_id)
        self._plotter.remove_actor(tracked.edges_actor)
        if tracked.faces_actor is not None:
            self._plotter.remove_actor(tracked.faces_actor)
        if tracked.origin_actor is not None:
            self._plotter.remove_actor(tracked.origin_actor)

    # =========================================================================
    # Display Control
    # =========================================================================

    def camera(self, position=None, focal_point=None):
        """Set camera position and/or focal point."""
        self._ensure_plotter()
        if position is not None:
            self._plotter.camera.position = position
        if focal_point is not None:
            self._plotter.camera.focal_point = focal_point

    def render(self):
        """Render the current frame."""
        if self._plotter is not None:
            self._plotter.render()

    def show(self):
        """Show the window (non-blocking)."""
        self._ensure_plotter()
        self._plotter.show(interactive_update=True, auto_close=False)

    def close(self):
        """Close the renderer window."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None

    @property
    def plotter(self) -> pv.Plotter | None:
        """Access the underlying PyVista plotter (for advanced use)."""
        return self._plotter
