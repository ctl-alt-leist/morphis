"""
Animated Canvas - Pure Renderer

Canvas that renders blades and updates on demand. The canvas does NO
transformation math - it just reads blade.visual_transform and renders.

Key class:
- AnimatedCanvas: Canvas that renders tracked blades

Usage:
    from morphis.core.rotations import rotate_blade

    canvas = AnimatedCanvas()
    canvas.track(blade)
    canvas.show()

    for frame in range(n_frames):
        rotate_blade(blade, axis, angle)  # External - modifies blade
        canvas.update()  # Re-reads blade.visual_transform and renders
"""

import weakref
from typing import Dict, Optional, Tuple

import pyvista as pv
import vtk
from numpy import array, pi, zeros
from numpy.linalg import norm

from morphis.ga.model import Blade
from morphis.visualization.theme import Color, get_theme


def _trivector_corners(origin, u, v, w):
    """Compute 8 corners of a parallelepiped."""
    return [
        origin,
        origin + u,
        origin + v,
        origin + w,
        origin + u + v,
        origin + u + w,
        origin + v + w,
        origin + u + v + w,
    ]


class TrackedBlade:
    """Internal tracking state for a blade."""

    def __init__(
        self,
        blade: Blade,
        edges_mesh,
        faces_mesh,
        edges_actor,
        faces_actor,
        color: Color,
    ):
        self.blade_ref = weakref.ref(blade)
        self.edges_mesh = edges_mesh
        self.faces_mesh = faces_mesh
        self.edges_actor = edges_actor
        self.faces_actor = faces_actor
        self.color = color
        self.vtk_transform = vtk.vtkTransform()

        # Apply transform to actors
        self.edges_actor.SetUserTransform(self.vtk_transform)
        self.faces_actor.SetUserTransform(self.vtk_transform)

    def sync_from_blade(self):
        """Read blade.visual_transform and update VTK transform."""
        blade = self.blade_ref()
        if blade is None:
            return

        vt = blade.visual_transform
        self.vtk_transform.Identity()

        # Apply translation first (will be applied second to points in VTK)
        self.vtk_transform.Translate(vt.translation[0], vt.translation[1], vt.translation[2])

        # Apply rotation second (will be applied first to points in VTK)
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(vt.rotation)
        rotvec = rot.as_rotvec()
        angle_rad = norm(rotvec)

        if angle_rad > 1e-10:
            axis = rotvec / angle_rad
            angle_deg = angle_rad * 180.0 / pi
            self.vtk_transform.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])


class AnimatedCanvas:
    """
    Canvas that renders tracked blades.

    The canvas does NO transformation logic. It just:
    1. Tracks blades (stores references, creates VTK actors)
    2. On update(), reads each blade's visual_transform and renders

    Example:
        from morphis.core.rotations import rotate_blade

        canvas = AnimatedCanvas()
        canvas.track(my_trivector)
        canvas.show()

        for t in range(100):
            rotate_blade(my_trivector, axis=(1,1,1), angle=0.1)
            canvas.update()
    """

    def __init__(
        self,
        theme: str = "obsidian",
        size: Tuple[int, int] = (1200, 900),
        show_basis: bool = True,
    ):
        if isinstance(theme, str):
            theme = get_theme(theme)

        self.theme = theme
        self._size = size
        self._show_basis = show_basis
        self._tracked: Dict[int, TrackedBlade] = {}
        self._color_index = 0
        self._plotter = None
        self._recording = False

    def _ensure_plotter(self):
        """Create plotter if not yet created."""
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=False)
            self._plotter.set_background(self.theme.background)
            self._plotter.window_size = self._size

            if self._show_basis:
                from morphis.visualization.drawing import draw_coordinate_basis

                axis_color = (0.85, 0.85, 0.85) if not self._is_light_theme() else (0.15, 0.15, 0.15)
                draw_coordinate_basis(self._plotter, color=axis_color)

    def _is_light_theme(self) -> bool:
        r, g, b = self.theme.background
        return 0.299 * r + 0.587 * g + 0.114 * b > 0.5

    def _next_color(self) -> Color:
        color = self.theme.palette[self._color_index % len(self.theme.palette)]
        self._color_index += 1
        return color

    def _create_parallelepiped_mesh(self, corners, edge_radius=0.008):
        """Create mesh for parallelepiped with given corners."""
        edge_indices = [
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

        edge_meshes = []
        for i, j in edge_indices:
            line = pv.Line(corners[i], corners[j])
            tube = line.tube(radius=edge_radius)
            edge_meshes.append(tube)

        combined_edges = edge_meshes[0]
        for mesh in edge_meshes[1:]:
            combined_edges = combined_edges.merge(mesh)

        face_indices = [
            [0, 1, 4, 2],
            [3, 5, 7, 6],
            [0, 1, 5, 3],
            [2, 4, 7, 6],
            [0, 2, 6, 3],
            [1, 4, 7, 5],
        ]

        face_meshes = []
        for indices in face_indices:
            quad = pv.Quadrilateral([corners[k] for k in indices])
            face_meshes.append(quad)

        combined_faces = face_meshes[0]
        for mesh in face_meshes[1:]:
            combined_faces = combined_faces.merge(mesh)

        return combined_edges, combined_faces

    def track(self, blade: Blade, color: Optional[Color] = None) -> int:
        """
        Start tracking a blade for rendering.

        Args:
            blade: The blade to track
            color: Optional color override

        Returns:
            blade_id: ID for this tracked blade
        """
        self._ensure_plotter()

        blade_id = id(blade)
        if blade_id in self._tracked:
            return blade_id

        if color is None:
            color = self._next_color()

        if blade.grade == 3:
            # Create unit parallelepiped
            origin = zeros(3)
            u = array([1.0, 0.0, 0.0])
            v = array([0.0, 1.0, 0.0])
            w = array([0.0, 0.0, 1.0])
            corners = _trivector_corners(origin, u, v, w)
            edges_mesh, faces_mesh = self._create_parallelepiped_mesh(corners)

            # Add to plotter and get actors
            edges_actor = self._plotter.add_mesh(edges_mesh, color=color, smooth_shading=True)
            faces_actor = self._plotter.add_mesh(faces_mesh, color=color, opacity=0.2, smooth_shading=True)

            # Create tracking record
            tracked = TrackedBlade(blade, edges_mesh, faces_mesh, edges_actor, faces_actor, color)
            self._tracked[blade_id] = tracked

        else:
            raise NotImplementedError(f"Tracking grade {blade.grade} blades not yet implemented")

        return blade_id

    def update(self):
        """
        Update the display by reading all tracked blades' visual transforms.

        This is the main rendering call. It:
        1. Reads blade.visual_transform for each tracked blade
        2. Applies transforms to VTK actors
        3. Renders the scene
        """
        for tracked in self._tracked.values():
            tracked.sync_from_blade()

        self._render()

    def _render(self):
        """Render the current frame."""
        if self._plotter is not None:
            self._plotter.render()

            if self._recording:
                self._plotter.write_frame()

    def camera(self, position=None, focal_point=None):
        """Set camera position."""
        self._ensure_plotter()
        if position is not None:
            self._plotter.camera.position = position
        if focal_point is not None:
            self._plotter.camera.focal_point = focal_point

    def show(self, interactive: bool = True):
        """Display the canvas."""
        self._ensure_plotter()
        if interactive:
            self._plotter.show(interactive_update=True, auto_close=False)
        else:
            self._plotter.show(interactive_update=True, auto_close=False)

    def start_recording(self, filename: str, fps: int = 60):
        """Start recording frames to a video file."""
        self._ensure_plotter()
        self._plotter.open_movie(filename, framerate=fps, quality=9)
        self._recording = True

    def stop_recording(self):
        """Stop recording and save the video."""
        if self._recording:
            self._plotter.close()
            self._recording = False

    def close(self):
        """Close the canvas."""
        if self._recording:
            self.stop_recording()
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
