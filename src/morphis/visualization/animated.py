"""
Animated Canvas - Pure Renderer

Canvas that renders blades and updates on demand. The canvas reads blade.data
directly and computes visual representations. No transformation state is stored
on the blade objects themselves.

Key classes:
- AnimatedCanvas: Canvas that renders tracked blades
- Timeline integration via play() method

Usage:
    # Manual control
    canvas = AnimatedCanvas()
    canvas.track(blade)
    canvas.show()
    for frame in range(n_frames):
        blade.data[...] = transformed_data
        canvas.update()

    # Timeline-based
    canvas = AnimatedCanvas()
    canvas.play(timeline)  # Runs the full animation
"""

import os
import subprocess
import sys
import time as time_module
import weakref
from typing import TYPE_CHECKING

import pyvista as pv
from numpy import array, copy as np_copy, zeros
from numpy.typing import NDArray

from morphis.ga.model import Blade
from morphis.ga.motors import Motor
from morphis.visualization.drawing import factor_blade
from morphis.visualization.theme import Color, get_theme


if TYPE_CHECKING:
    from morphis.visualization.sequence import Timeline


def _bring_window_to_front():
    """Bring the current application window to front (macOS only)."""
    if sys.platform == "darwin":
        try:
            # Use AppKit for more reliable window focusing on macOS
            from AppKit import NSApp, NSApplication

            NSApplication.sharedApplication()
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            # Fallback to AppleScript if AppKit not available
            script = f"""
            tell application "System Events"
                set frontmost of the first process whose unix id is {os.getpid()} to true
            end tell
            """
            subprocess.run(["osascript", "-e", script], capture_output=True)


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
        edges_actor,
        faces_actor,
        color: Color,
        plotter: pv.Plotter,
        vectors: NDArray | None = None,
        origin: NDArray | None = None,
    ):
        self.blade_ref = weakref.ref(blade)
        self.edges_actor = edges_actor
        self.faces_actor = faces_actor
        self.color = color
        self.plotter = plotter
        # Store spanning vectors directly for animation
        # This avoids the issue of factor_blade always returning axis-aligned vectors
        self.vectors = vectors
        # Track origin position for translation
        self.origin = origin if origin is not None else zeros(3)

    def sync_from_vectors(self):
        """Update VTK meshes from stored spanning vectors."""
        if self.vectors is None:
            return

        blade = self.blade_ref()
        if blade is None:
            return

        if blade.grade == 1:
            # Vector: single direction
            direction = self.vectors[0]
            edges_mesh = self._create_arrow_mesh(self.origin, direction)
            self.edges_actor.mapper.SetInputData(edges_mesh)
            # No faces for vectors
            if self.faces_actor is not None:
                self.faces_actor.GetProperty().SetOpacity(0)

        elif blade.grade == 2:
            # Bivector: two spanning vectors
            u, v = self.vectors[0], self.vectors[1]
            edges_mesh, faces_mesh = self._create_parallelogram_mesh(self.origin, u[:3], v[:3])
            self.edges_actor.mapper.SetInputData(edges_mesh)
            if self.faces_actor is not None:
                self.faces_actor.mapper.SetInputData(faces_mesh)

        elif blade.grade == 3:
            u, v, w = self.vectors[0], self.vectors[1], self.vectors[2]
            corners = _trivector_corners(self.origin, u[:3], v[:3], w[:3])
            edges_mesh, faces_mesh = self._create_parallelepiped_mesh(corners)
            self.edges_actor.mapper.SetInputData(edges_mesh)
            self.faces_actor.mapper.SetInputData(faces_mesh)

    def sync_from_blade(self):
        """Read blade.data and update the VTK meshes."""
        blade = self.blade_ref()
        if blade is None:
            return

        if blade.grade == 3:
            self._sync_trivector(blade)

    def _sync_trivector(self, blade: Blade):
        """Update parallelepiped mesh from trivector data."""
        # Factor the trivector to get spanning vectors
        u, v, w = factor_blade(blade)

        # Ensure 3D (pad with zeros if necessary)
        if len(u) < 3:
            u = array([*u, *[0.0] * (3 - len(u))])
        if len(v) < 3:
            v = array([*v, *[0.0] * (3 - len(v))])
        if len(w) < 3:
            w = array([*w, *[0.0] * (3 - len(w))])

        # Store vectors for animation
        self.vectors = array([u[:3], v[:3], w[:3]])

        origin = zeros(3)
        corners = _trivector_corners(origin, u[:3], v[:3], w[:3])

        # Rebuild meshes with new geometry
        edges_mesh, faces_mesh = self._create_parallelepiped_mesh(corners)

        # Update the actors with new meshes
        self.edges_actor.mapper.SetInputData(edges_mesh)
        self.faces_actor.mapper.SetInputData(faces_mesh)

    def _create_arrow_mesh(self, origin, direction, shaft_radius=0.008, tip_ratio=0.12, tip_radius_ratio=2.5):
        """Create mesh for an arrow from origin in given direction."""
        from numpy.linalg import norm

        length = norm(direction)
        if length < 1e-10:
            # Return a tiny sphere for zero-length vector
            return pv.Sphere(radius=shaft_radius, center=origin)

        dir_norm = direction / length
        tip_length = length * tip_ratio
        shaft_length = length - tip_length
        tip_radius = shaft_radius * tip_radius_ratio

        shaft_end = origin + dir_norm * shaft_length

        # Shaft cylinder
        shaft = pv.Cylinder(
            center=(origin + shaft_end) / 2,
            direction=dir_norm,
            radius=shaft_radius,
            height=shaft_length,
            resolution=20,
            capping=True,
        )

        # Tip cone
        tip = pv.Cone(
            center=shaft_end + dir_norm * (tip_length / 2),
            direction=dir_norm,
            height=tip_length,
            radius=tip_radius,
            resolution=20,
            capping=True,
        )

        return shaft.merge(tip)

    def _create_parallelogram_mesh(self, origin, u, v, edge_radius=0.006):
        """Create mesh for a parallelogram defined by vectors u and v from origin."""
        corners = [
            origin,
            origin + u,
            origin + u + v,
            origin + v,
        ]

        # Create edges
        edge_meshes = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            line = pv.Line(start, end)
            tube = line.tube(radius=edge_radius)
            edge_meshes.append(tube)

        combined_edges = edge_meshes[0]
        for mesh in edge_meshes[1:]:
            combined_edges = combined_edges.merge(mesh)

        # Create face
        quad = pv.Quadrilateral(corners)

        return combined_edges, quad

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


class AnimatedCanvas:
    """
    Canvas that renders tracked blades.

    The canvas reads blade.data directly and handles all visualization logic.
    Blades are pure algebraic objects with no visualization state.

    Example:
        canvas = AnimatedCanvas()
        canvas.track(my_trivector)
        canvas.show()

        for t in range(100):
            # Transform blade data in-place
            my_trivector.data[...] = transformed.data
            canvas.update()
    """

    def __init__(
        self,
        theme: str = "obsidian",
        size: tuple[int, int] = (1800, 1350),
        show_basis: bool = True,
    ):
        if isinstance(theme, str):
            theme = get_theme(theme)

        self.theme = theme
        self._size = size
        self._show_basis = show_basis
        self._tracked: dict[int, TrackedBlade] = {}
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

                # Use theme's computed axis color for proper contrast
                draw_coordinate_basis(self._plotter, color=self.theme.axis_color)

    def _next_color(self) -> Color:
        color = self.theme.palette[self._color_index % len(self.theme.palette)]
        self._color_index += 1
        return color

    def track(self, blade: Blade, color: Color | None = None) -> int:
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

        if blade.grade == 1:
            # Vector: single direction
            direction = blade.data.copy()
            if len(direction) < 3:
                direction = array([*direction, *[0.0] * (3 - len(direction))])
            direction = direction[:3]

            origin = zeros(3)

            # Create initial arrow mesh
            tracked_temp = TrackedBlade.__new__(TrackedBlade)
            edges_mesh = tracked_temp._create_arrow_mesh(origin, direction)

            # Add to plotter - vectors have no faces
            edges_actor = self._plotter.add_mesh(edges_mesh, color=color, smooth_shading=True)
            faces_actor = None

            # Create tracking record
            initial_vectors = array([direction])
            tracked = TrackedBlade(blade, edges_actor, faces_actor, color, self._plotter, vectors=initial_vectors)
            self._tracked[blade_id] = tracked

        elif blade.grade == 2:
            # Bivector: factor into two spanning vectors
            u, v = factor_blade(blade)

            # Ensure 3D
            if len(u) < 3:
                u = array([*u, *[0.0] * (3 - len(u))])
            if len(v) < 3:
                v = array([*v, *[0.0] * (3 - len(v))])

            origin = zeros(3)

            # Create initial parallelogram mesh
            tracked_temp = TrackedBlade.__new__(TrackedBlade)
            edges_mesh, faces_mesh = tracked_temp._create_parallelogram_mesh(origin, u[:3], v[:3])

            # Add to plotter
            edges_actor = self._plotter.add_mesh(edges_mesh, color=color, smooth_shading=True)
            faces_actor = self._plotter.add_mesh(faces_mesh, color=color, opacity=0.25, smooth_shading=True)

            # Create tracking record
            initial_vectors = array([u[:3], v[:3]])
            tracked = TrackedBlade(blade, edges_actor, faces_actor, color, self._plotter, vectors=initial_vectors)
            self._tracked[blade_id] = tracked

        elif blade.grade == 3:
            # Factor the trivector to get initial geometry
            u, v, w = factor_blade(blade)

            # Ensure 3D
            if len(u) < 3:
                u = array([*u, *[0.0] * (3 - len(u))])
            if len(v) < 3:
                v = array([*v, *[0.0] * (3 - len(v))])
            if len(w) < 3:
                w = array([*w, *[0.0] * (3 - len(w))])

            origin = zeros(3)
            corners = _trivector_corners(origin, u[:3], v[:3], w[:3])

            # Create initial meshes
            tracked_temp = TrackedBlade.__new__(TrackedBlade)
            edges_mesh, faces_mesh = tracked_temp._create_parallelepiped_mesh(corners)

            # Add to plotter and get actors
            edges_actor = self._plotter.add_mesh(edges_mesh, color=color, smooth_shading=True)
            faces_actor = self._plotter.add_mesh(faces_mesh, color=color, opacity=0.2, smooth_shading=True)

            # Create tracking record with initial vectors
            initial_vectors = array([u[:3], v[:3], w[:3]])
            tracked = TrackedBlade(blade, edges_actor, faces_actor, color, self._plotter, vectors=initial_vectors)
            self._tracked[blade_id] = tracked

        else:
            raise NotImplementedError(f"Tracking grade {blade.grade} blades not yet implemented")

        return blade_id

    def update(self):
        """
        Update the display by reading all tracked blades' data.

        This is the main rendering call. It:
        1. Reads blade.data for each tracked blade
        2. Computes visual geometry from the data
        3. Updates VTK actors
        4. Renders the scene
        """
        for tracked in self._tracked.values():
            tracked.sync_from_blade()

        self._render()

    def _update_from_vectors(self):
        """
        Update display from stored vectors (used during animation).

        Unlike update(), this uses the pre-computed vectors stored in
        TrackedBlade rather than factoring the blade tensor.
        """
        for tracked in self._tracked.values():
            tracked.sync_from_vectors()

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
        _bring_window_to_front()

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

    def play(
        self,
        timeline: "Timeline",
        fps: int = 60,
        blocking: bool = True,
        filename: str | None = None,
        follow: Blade | None = None,
    ):
        """
        Play a timeline animation.

        This method:
        1. Tracks all blades in the timeline
        2. Stores their initial state
        3. Runs the animation loop, applying actions at each frame
        4. Renders and optionally records

        Args:
            timeline: The Timeline to play
            fps: Frames per second
            blocking: If True, blocks until animation completes
            filename: Optional video file to record to
            follow: Optional blade to follow with camera (focal point tracks center)

        Example:
            canvas = AnimatedCanvas(theme="obsidian")
            canvas.play(timeline)  # Blocking by default

            # Non-blocking with recording
            canvas.play(timeline, blocking=False, filename="output.mp4")

            # Camera follows trivector as it translates
            canvas.play(timeline, follow=q)
        """

        self._ensure_plotter()

        # Track all blades and store their initial vectors and origins
        vectors_initial: dict[int, array] = {}
        origins_initial: dict[int, array] = {}
        blade_visible: dict[int, bool] = {}
        blade_opacity: dict[int, float] = {}

        # Extract blade colors from DrawActions and ReplaceActions
        from morphis.visualization.sequence import DrawAction, ReplaceAction

        blade_colors: dict[int, Color] = {}
        for action in timeline.actions:
            if isinstance(action, DrawAction) and action.color is not None:
                blade_colors[id(action.blade)] = action.color
            elif isinstance(action, ReplaceAction) and action.new_color is not None:
                blade_colors[id(action.new_blade)] = action.new_color

        for blade in timeline.all_blades():
            blade_id = id(blade)
            blade_visible[blade_id] = False
            blade_opacity[blade_id] = 0.0
            # Use color from DrawAction if specified
            color = blade_colors.get(blade_id)
            self.track(blade, color=color)
            # Store initial vectors and origin from the tracked blade
            vectors_initial[blade_id] = np_copy(self._tracked[blade_id].vectors)
            origins_initial[blade_id] = np_copy(self._tracked[blade_id].origin)
            # Start invisible
            self._set_blade_opacity(blade, 0.0)

        total_duration = timeline.total_duration

        # Recording
        if filename:
            self.start_recording(filename, fps=fps)
            self._plotter.off_screen = True

        # Show the window
        if not filename:
            self.show()

        # Animation loop
        start_time = time_module.time()
        n_frames = int(total_duration * fps)

        # Store initial camera offset if following a blade
        follow_id = id(follow) if follow is not None else None
        camera_offset = None
        if follow_id is not None and follow_id in self._tracked:
            initial_center = self._get_blade_center(follow_id)
            camera_offset = array(self._plotter.camera.position) - initial_center

        for frame in range(n_frames + 1):
            t = frame / fps

            # Process each action up to current time
            self._apply_timeline_state(timeline, t, vectors_initial, origins_initial, blade_visible, blade_opacity)

            # Update camera to follow blade
            if follow_id is not None and follow_id in self._tracked and camera_offset is not None:
                center = self._get_blade_center(follow_id)
                self._plotter.camera.focal_point = center
                self._plotter.camera.position = center + camera_offset

            # Render - use vector-based sync for animation
            self._update_from_vectors()

            # Timing (real-time playback if not recording)
            if not filename and blocking:
                elapsed = time_module.time() - start_time
                target = frame / fps
                if target > elapsed:
                    time_module.sleep(target - elapsed)

        # Stop recording if active
        if filename:
            self.stop_recording()
            print(f"Saved animation to {filename}")

        # Keep window open after animation
        if not filename and blocking:
            print("Animation complete. Close window to exit.")
            self._plotter.iren.interactor.Start()

    def _set_blade_opacity(self, blade: Blade, opacity: float):
        """Set the opacity of a tracked blade's actors."""
        blade_id = id(blade)
        if blade_id in self._tracked:
            tracked = self._tracked[blade_id]
            tracked.edges_actor.GetProperty().SetOpacity(opacity)
            if tracked.faces_actor is not None:
                tracked.faces_actor.GetProperty().SetOpacity(opacity * 0.2)

    def _apply_timeline_state(
        self,
        timeline: "Timeline",
        time: float,
        vectors_initial: dict,
        origins_initial: dict,
        blade_visible: dict,
        blade_opacity: dict,
    ):
        """Apply the timeline state at a given time to all blades."""
        from morphis.visualization.sequence import (
            DrawAction,
            HideAction,
            ReplaceAction,
            RotateAction,
            RotateToPlaneAction,
            ScrewAction,
        )

        # Reset all tracked blades to initial vectors and origins
        for blade_id, initial_vectors in vectors_initial.items():
            if blade_id in self._tracked:
                self._tracked[blade_id].vectors = np_copy(initial_vectors)
                self._tracked[blade_id].origin = np_copy(origins_initial[blade_id])

        # Process all actions up to current time
        elapsed = 0.0
        for action in timeline.actions:
            action_end = elapsed + action.duration
            local_time = time - elapsed

            if time < elapsed:
                # Haven't reached this action yet
                break

            if isinstance(action, DrawAction):
                blade_id = id(action.blade)
                if time >= action_end:
                    blade_visible[blade_id] = True
                    blade_opacity[blade_id] = 1.0
                else:
                    blade_visible[blade_id] = True
                    blade_opacity[blade_id] = action.opacity_at(local_time)
                self._set_blade_opacity(action.blade, blade_opacity[blade_id])

            elif isinstance(action, HideAction):
                blade_id = id(action.blade)
                if time >= action_end:
                    blade_visible[blade_id] = False
                    blade_opacity[blade_id] = 0.0
                else:
                    blade_opacity[blade_id] = action.opacity_at(local_time)
                self._set_blade_opacity(action.blade, blade_opacity[blade_id])

            elif isinstance(action, ReplaceAction):
                old_id = id(action.old_blade)
                new_id = id(action.new_blade)
                if time >= action_end:
                    blade_visible[old_id] = False
                    blade_opacity[old_id] = 0.0
                    blade_visible[new_id] = True
                    blade_opacity[new_id] = 1.0
                else:
                    blade_opacity[old_id] = action.old_opacity_at(local_time)
                    blade_opacity[new_id] = action.new_opacity_at(local_time)
                self._set_blade_opacity(action.old_blade, blade_opacity[old_id])
                self._set_blade_opacity(action.new_blade, blade_opacity[new_id])

            elif isinstance(action, RotateAction):
                blade = action.blade
                blade_id = id(blade)

                if time >= action_end:
                    angle = action.angle
                else:
                    angle = action.angle_at(local_time)

                # Apply rotation via Motor.rotor with the bivector plane
                B_pga = self._embed_bivector_to_pga(action.plane)
                motor = Motor.rotor(B_pga, angle)

                if action.center is not None:
                    T_to = Motor.translator(-action.center)
                    T_back = Motor.translator(action.center)
                    motor = T_back.compose(motor).compose(T_to)

                # Transform current vectors (builds on previous actions)
                current_vectors = self._tracked[blade_id].vectors
                transformed = motor.apply_to_euclidean(current_vectors)
                self._tracked[blade_id].vectors = transformed

                # Also rotate the origin to make the object orbit around the rotation center
                current_origin = self._tracked[blade_id].origin
                if current_origin is not None and (current_origin != 0).any():
                    # Rotate the origin point around the world origin (or specified center)
                    origin_transformed = motor.apply_to_euclidean(current_origin.reshape(1, 3))
                    self._tracked[blade_id].origin = origin_transformed.flatten()

            elif isinstance(action, ScrewAction):
                blade = action.blade
                blade_id = id(blade)

                if time >= action_end:
                    progress = 1.0
                else:
                    progress = action.progress_at(local_time)

                # Apply screw motion: rotation via Motor + direct translation
                # (PGA translation via sandwich product is known to have limitations,
                # so we apply translation directly to the origin)
                B_pga = self._embed_bivector_to_pga(action.plane)
                rotor = Motor.rotor(B_pga, action.angle * progress)

                # Handle center offset if specified
                current_vectors = self._tracked[blade_id].vectors
                if action.center is not None:
                    # Translate to origin, rotate, translate back
                    T_to = Motor.translator(-action.center)
                    T_back = Motor.translator(action.center)
                    motor = T_back.compose(rotor).compose(T_to)
                    transformed = motor.apply_to_euclidean(current_vectors)
                else:
                    transformed = rotor.apply_to_euclidean(current_vectors)

                self._tracked[blade_id].vectors = transformed
                # Apply translation directly to the origin (bypasses PGA limitation)
                self._tracked[blade_id].origin = self._tracked[blade_id].origin + action.translation * progress

            elif isinstance(action, RotateToPlaneAction):
                blade = action.blade
                blade_id = id(blade)

                if time >= action_end:
                    progress = 1.0
                else:
                    progress = action.progress_at(local_time)

                # Get current vectors and compute current normal
                current_vectors = self._tracked[blade_id].vectors
                if blade.grade == 3:
                    # For trivector: compute normal from cross product of two spanning vectors
                    u, v = current_vectors[0], current_vectors[1]
                    from numpy import cross
                    from numpy.linalg import norm

                    current_normal = cross(u[:3], v[:3])
                    n_len = norm(current_normal)
                    if n_len > 1e-10:
                        current_normal = current_normal / n_len
                    else:
                        current_normal = array([0.0, 0.0, 1.0])
                else:
                    # For other grades, use first vector as normal approximation
                    current_normal = current_vectors[0][:3]
                    n_len = norm(current_normal)
                    if n_len > 1e-10:
                        current_normal = current_normal / n_len

                # Target normal
                target = array(action.target_normal, dtype=float)
                target = target / norm(target)

                # Compute rotation axis and angle using cross and dot products
                from numpy import arccos, clip, dot

                axis = cross(current_normal, target)
                axis_len = norm(axis)

                if axis_len > 1e-10:
                    axis = axis / axis_len
                    cos_angle = clip(dot(current_normal, target), -1.0, 1.0)
                    total_angle = arccos(cos_angle)

                    # Apply partial rotation based on progress
                    angle = total_angle * progress

                    # Build bivector from axis (dual in 3D)
                    # Axis (a, b, c) -> bivector a*e23 + b*e31 + c*e12
                    from morphis.ga.constructors import basis_vectors

                    e1, e2, e3 = basis_vectors(dim=3)
                    e23 = e2 ^ e3
                    e31 = e3 ^ e1
                    e12_plane = e1 ^ e2
                    rotation_plane = (e23 * axis[0]) + (e31 * axis[1]) + (e12_plane * axis[2])

                    # Embed to PGA and apply rotation
                    B_pga = self._embed_bivector_to_pga(rotation_plane)
                    motor = Motor.rotor(B_pga, angle)
                    transformed = motor.apply_to_euclidean(current_vectors)
                    self._tracked[blade_id].vectors = transformed

            elapsed = action_end

    def _get_blade_by_id(self, blade_id: int) -> Blade | None:
        """Get a blade object by its id."""
        if blade_id in self._tracked:
            return self._tracked[blade_id].blade_ref()
        return None

    def _get_blade_center(self, blade_id: int) -> array:
        """
        Get the geometric center of a tracked blade.

        For trivectors, the center is origin + half the diagonal of the parallelepiped.
        """
        if blade_id not in self._tracked:
            return zeros(3)

        tracked = self._tracked[blade_id]
        origin = tracked.origin
        vectors = tracked.vectors

        if vectors is None:
            return origin

        # Center of parallelepiped: origin + 0.5 * (u + v + w)
        return origin + 0.5 * (vectors[0] + vectors[1] + vectors[2])

    def _blade_to_vectors(self, blade: Blade) -> array:
        """
        Extract the spanning vectors from a blade for transformation.

        For trivectors, returns the 3 spanning vectors.
        """
        if blade.grade == 3:
            u, v, w = factor_blade(blade)
            # Ensure 3D
            if len(u) < 3:
                u = array([*u, *[0.0] * (3 - len(u))])
            if len(v) < 3:
                v = array([*v, *[0.0] * (3 - len(v))])
            if len(w) < 3:
                w = array([*w, *[0.0] * (3 - len(w))])
            return array([u[:3], v[:3], w[:3]])
        else:
            raise NotImplementedError(f"Grade {blade.grade} not yet supported")

    def _vectors_to_blade(self, blade: Blade, vectors: array):
        """
        Update blade data from transformed spanning vectors.

        Reconstructs the antisymmetric blade tensor from the vectors.
        """
        if blade.grade == 3:
            u, v, w = vectors[0], vectors[1], vectors[2]
            # Reconstruct trivector via wedge (antisymmetrized outer product)
            dim = blade.dim
            data = zeros((dim, dim, dim))
            for i in range(min(3, dim)):
                for j in range(min(3, dim)):
                    for k in range(min(3, dim)):
                        # Antisymmetrized: u_i v_j w_k - u_i v_k w_j + ...
                        data[i, j, k] = (
                            u[i] * (v[j] * w[k] - v[k] * w[j])
                            + u[j] * (v[k] * w[i] - v[i] * w[k])
                            + u[k] * (v[i] * w[j] - v[j] * w[i])
                        )
            blade.data[...] = data
        else:
            raise NotImplementedError(f"Grade {blade.grade} not yet supported")

    def _embed_bivector_to_pga(self, B: Blade) -> Blade:
        """
        Embed a 3D Euclidean bivector into 4D PGA.

        Takes a bivector in 3D (dim=3) and embeds it into the Euclidean
        subspace of 4D PGA (dim=4), with indices shifted by 1 to account
        for the ideal (e0) direction.

        Args:
            B: Grade-2 blade in 3D Euclidean space

        Returns:
            Grade-2 blade in 4D PGA (indices 1,2,3)
        """
        if B.grade != 2:
            raise ValueError(f"Expected bivector (grade 2), got grade {B.grade}")

        # Embed into PGA: shift indices by 1
        # B[i,j] in 3D -> B_pga[i+1, j+1] in 4D
        dim_euclidean = B.dim
        dim_pga = dim_euclidean + 1

        B_pga_data = zeros((dim_pga, dim_pga))
        for i in range(dim_euclidean):
            for j in range(dim_euclidean):
                B_pga_data[i + 1, j + 1] = B.data[i, j]

        return Blade(data=B_pga_data, grade=2, dim=dim_pga, cdim=0)
