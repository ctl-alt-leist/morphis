"""
Renderer - Low-level visualization layer

The Renderer manages VTK/PyVista actors and tracks objects that can be
dynamically updated. It delegates all mesh creation to drawing.py.

It has no knowledge of:
- Geometric algebra (blades, motors, etc.)
- Time or animation
- Effects or scheduling
"""

from typing import Any

import pyvista as pv
from numpy import array
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.visuals.drawing.vectors import create_blade_mesh, create_frame_mesh, draw_coordinate_basis
from morphis.visuals.theme import Color, Theme, get_theme


class TrackedObject(BaseModel):
    """Internal state for a rendered object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    obj_id: int
    grade: int
    color: Color
    edges_actor: Any  # pv.Actor - using Any for PyVista compatibility
    faces_actor: Any | None  # pv.Actor | None
    origin_actor: Any | None  # pv.Actor | None
    opacity: float = 1.0
    projection_axes: tuple[int, int, int] | None = None  # For nD -> 3D projection
    filled: bool = False  # For frames: whether to show edges and faces


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
        self._basis_label_actors: list | None = None
        self._current_basis_labels: tuple[str, str, str] | None = None

    def _ensure_plotter(self):
        """Create plotter if not yet created."""
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=False)
            self._plotter.set_background(self.theme.background)
            self._plotter.window_size = self._size

            if self._show_basis:
                self._basis_label_actors = draw_coordinate_basis(
                    self._plotter,
                    color=self.theme.axis_color,
                    labels=self._current_basis_labels,
                )

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
        projection_axes: tuple[int, int, int] | None = None,
        filled: bool = False,
    ):
        """
        Add a new object to the renderer.

        Args:
            obj_id: Unique identifier for this object
            grade: Geometric grade (1=vector, 2=bivector, 3=trivector, 4=quadvector, -1=frame)
            origin: Origin point (nD)
            vectors: Spanning vectors (shape depends on grade)
            color: RGB color tuple (0-1 range), or None for auto
            opacity: Initial opacity [0, 1]
            projection_axes: For nD blades, which 3 axes to project onto
            filled: For frames, whether to show edges and faces of spanned shape
        """
        self._ensure_plotter()

        if obj_id in self._objects:
            self.update_object(obj_id, origin, vectors, opacity, projection_axes)
            return

        if color is None:
            color = self._next_color()

        origin = array(origin, dtype=float)
        vectors = array(vectors, dtype=float)

        # Create meshes using centralized drawing functions
        if grade == -1:
            # Frame: k arrows from origin (optionally with edges/faces)
            edges_mesh, faces_mesh, origin_mesh = create_frame_mesh(
                origin, vectors, projection_axes=projection_axes, filled=filled
            )
        else:
            # Vector: sequential construction
            edges_mesh, faces_mesh, origin_mesh = create_blade_mesh(
                grade, origin, vectors, projection_axes=projection_axes
            )

        # Add to plotter
        edges_actor = None
        if edges_mesh is not None:
            edges_actor = self._plotter.add_mesh(edges_mesh, color=color, opacity=opacity, smooth_shading=True)

        faces_actor = None
        if faces_mesh is not None:
            if grade == 2:
                face_opacity = opacity * 0.25
            elif grade == 3:
                face_opacity = opacity * 0.2
            elif grade == 4:
                face_opacity = opacity * 0.15
            elif grade == -1:
                face_opacity = opacity * 0.2  # Frame faces
            else:
                face_opacity = opacity * 0.2
            faces_actor = self._plotter.add_mesh(faces_mesh, color=color, opacity=face_opacity, smooth_shading=True)

        origin_actor = self._plotter.add_mesh(origin_mesh, color=color, opacity=opacity, smooth_shading=True)

        self._objects[obj_id] = TrackedObject(
            obj_id=obj_id,
            grade=grade,
            color=color,
            edges_actor=edges_actor,
            faces_actor=faces_actor,
            origin_actor=origin_actor,
            opacity=opacity,
            projection_axes=projection_axes,
            filled=filled,
        )

    def update_object(
        self,
        obj_id: int,
        origin: NDArray,
        vectors: NDArray,
        opacity: float | None = None,
        projection_axes: tuple[int, int, int] | None = None,
    ):
        """
        Update an existing object's geometry and/or opacity.

        Args:
            obj_id: The object to update
            origin: New origin point
            vectors: New spanning vectors
            opacity: New opacity (or None to keep current)
            projection_axes: For nD blades, which 3 axes to project onto
        """
        if obj_id not in self._objects:
            return

        tracked = self._objects[obj_id]
        origin = array(origin, dtype=float)
        vectors = array(vectors, dtype=float)

        if opacity is not None:
            tracked.opacity = opacity

        if projection_axes is not None:
            tracked.projection_axes = projection_axes

        # Recreate meshes using centralized drawing functions
        if tracked.grade == -1:
            # Frame: k arrows from origin (optionally with edges/faces)
            edges_mesh, faces_mesh, origin_mesh = create_frame_mesh(
                origin, vectors, projection_axes=tracked.projection_axes, filled=tracked.filled
            )
        else:
            # Vector: sequential construction
            edges_mesh, faces_mesh, origin_mesh = create_blade_mesh(
                tracked.grade, origin, vectors, projection_axes=tracked.projection_axes
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
            if tracked.grade == 2:
                face_opacity = tracked.opacity * 0.25
            elif tracked.grade == 3:
                face_opacity = tracked.opacity * 0.2
            elif tracked.grade == 4:
                face_opacity = tracked.opacity * 0.15
            else:
                face_opacity = tracked.opacity * 0.2
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

    def set_basis_labels(self, labels: tuple[str, str, str]):
        """
        Update coordinate basis labels.

        Args:
            labels: New labels for (x, y, z) axes, e.g. ("$e_1$", "$e_2$", "$e_3$")
        """
        if labels == self._current_basis_labels:
            return  # No change needed

        self._current_basis_labels = labels

        if self._plotter is None or not self._show_basis:
            return

        # Remove old label actors
        if self._basis_label_actors:
            for actor in self._basis_label_actors:
                self._plotter.remove_actor(actor)

        # Draw new labels only (arrows stay the same)
        self._basis_label_actors = self._draw_basis_labels(labels)

    def _draw_basis_labels(self, labels: tuple[str, str, str]) -> list:
        """Draw only the basis labels (not the arrows)."""
        from numpy import array
        from numpy.linalg import norm

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

        label_actors = []
        for direction, offset_dir, axis_name in zip(directions, offset_dirs, labels, strict=False):
            offset = offset_dir / norm(offset_dir) * label_offset
            label_pos = direction * 0.5 + offset
            actor = self._plotter.add_point_labels(
                [label_pos],
                [axis_name],
                font_size=12,
                text_color=self.theme.axis_color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )
            label_actors.append(actor)
        return label_actors

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
