"""
Animation - Observer and Recorder

The Animation class observes geometric objects (blades) and records their
state over time. It is completely ignorant of what transformations are
applied to the objects - it just reads their current state when asked.

Key concepts:
- track(blade): Register a blade to observe
- capture(t): Snapshot all tracked objects at time t
- play(): Play back recorded snapshots

The Animation coordinates between:
- User script (controls geometry and time)
- Effects (visual modifications like fades)
- Renderer (handles actual drawing)

Uses the Observer class internally for core tracking functionality.
"""

import sys
import time as time_module
from dataclasses import dataclass

from numpy import array, copy as np_copy, zeros
from numpy.typing import NDArray

from morphis.ga.model import Blade
from morphis.utils.observer import Observer
from morphis.visualization.effects import Effect, FadeIn, FadeOut, compute_opacity
from morphis.visualization.renderer import Renderer
from morphis.visualization.theme import Color, Theme


@dataclass
class Snapshot:
    """State of all tracked objects at a specific time."""

    t: float
    # obj_id -> (origin, vectors, opacity, projection_axes)
    states: dict[int, tuple[NDArray, NDArray, float, tuple[int, int, int] | None]]
    basis_labels: tuple[str, str, str] | None = None  # Optional basis labels for this frame


@dataclass
class AnimationTrack:
    """Animation-specific tracking info for a blade."""

    blade: Blade
    obj_id: int
    color: Color
    grade: int
    vectors: NDArray | None = None  # Override for spanning vectors
    origin: NDArray | None = None  # Override for origin
    projection_axes: tuple[int, int, int] | None = None  # For nD -> 3D projection


class Animation:
    """
    Animation observer and recorder.

    Observes blades and records their state over time. Supports both batch
    mode (record all, then play) and live mode (render as you go).

    Example (batch mode):
        anim = Animation(fps=60)
        anim.track(q, color=(0.85, 0.2, 0.2))
        anim.fade_in(q, t=0.0, duration=1.0)

        anim.start()
        for t in times:
            q.data[...] = transform(q)
            anim.capture(t)
        anim.play()

    Example (live mode):
        anim = Animation(fps=60)
        anim.track(q)

        anim.start(live=True)
        for t in times:
            q.data[...] = transform(q)
            anim.capture(t)
        anim.finish()
    """

    def __init__(
        self,
        fps: int = 60,
        theme: str | Theme = "obsidian",
        size: tuple[int, int] = (1800, 1350),
        show_basis: bool = True,
    ):
        self.fps = fps
        self._renderer = Renderer(theme=theme, size=size, show_basis=show_basis)
        self._observer = Observer()  # Core tracking via Observer
        self._tracks: dict[int, AnimationTrack] = {}  # Animation-specific data
        self._effects: list[Effect] = []
        self._snapshots: list[Snapshot] = []
        self._live = False
        self._started = False
        self._last_render_time: float | None = None
        self._start_wall_time: float | None = None

    # =========================================================================
    # Tracking
    # =========================================================================

    def track(self, *blades: Blade, color: Color | None = None) -> int | list[int]:
        """
        Register one or more blades to observe.

        Args:
            *blades: Blades to track
            color: Optional color override (applies to all)

        Returns:
            Object ID(s) for the blade(s)
        """
        ids = []
        for blade in blades:
            obj_id = id(blade)

            if obj_id in self._tracks:
                ids.append(obj_id)
                continue

            blade_color = color if color is not None else self._renderer._next_color()

            # Track in Observer for core functionality
            self._observer.track(blade)

            # Store animation-specific data
            self._tracks[obj_id] = AnimationTrack(
                blade=blade,
                obj_id=obj_id,
                color=blade_color,
                grade=blade.grade,
            )

            ids.append(obj_id)

        return ids[0] if len(ids) == 1 else ids

    def untrack(self, *blades: Blade):
        """Stop tracking one or more blades."""
        for blade in blades:
            obj_id = id(blade)
            if obj_id in self._tracks:
                del self._tracks[obj_id]
                self._observer.untrack(blade)
                self._renderer.remove_object(obj_id)

    def set_vectors(self, blade: Blade, vectors: NDArray, origin: NDArray | None = None):
        """
        Set the spanning vectors for a blade directly.

        This bypasses factor_blade() which cannot recover orientation from
        transformed blade data. Use this when you're transforming vectors
        directly and want the animation to show the transformation.

        Args:
            blade: The tracked blade
            vectors: Spanning vectors (shape depends on grade)
            origin: Optional origin point (default: [0,0,0])
        """
        obj_id = id(blade)
        if obj_id in self._tracks:
            self._tracks[obj_id].vectors = array(vectors, dtype=float)
            if origin is not None:
                self._tracks[obj_id].origin = array(origin, dtype=float)

    def set_projection(self, blade: Blade, axes: tuple[int, int, int]):
        """
        Set the projection axes for visualizing a high-dimensional blade.

        For blades in dimension > 3, this determines which 3 components
        are shown. For example:
        - (0, 1, 2) shows e1, e2, e3 (default)
        - (1, 2, 3) shows e2, e3, e4

        Args:
            blade: The tracked blade
            axes: Tuple of 3 axis indices to project onto
        """
        obj_id = id(blade)
        if obj_id in self._tracks:
            self._tracks[obj_id].projection_axes = axes

    def set_global_projection(self, axes: tuple[int, int, int]):
        """
        Set projection axes for all tracked blades.

        Convenience method for switching the view for all objects at once.

        Args:
            axes: Tuple of 3 axis indices to project onto
        """
        for track in self._tracks.values():
            track.projection_axes = axes

    # =========================================================================
    # Observer Delegation
    # =========================================================================

    @property
    def observer(self) -> Observer:
        """Access the underlying Observer for advanced use."""
        return self._observer

    def __len__(self) -> int:
        """Number of tracked objects."""
        return len(self._tracks)

    def __contains__(self, blade: Blade) -> bool:
        """Check if a blade is being tracked."""
        return id(blade) in self._tracks

    def __iter__(self):
        """Iterate over tracked blades."""
        return iter(track.blade for track in self._tracks.values())

    # =========================================================================
    # Effects
    # =========================================================================

    def fade_in(self, blade: Blade, t: float, duration: float):
        """
        Schedule a fade-in effect.

        Args:
            blade: The blade to fade in
            t: Start time (seconds)
            duration: Duration of fade (seconds)
        """
        obj_id = id(blade)
        self._effects.append(
            FadeIn(
                object_id=obj_id,
                t_start=t,
                t_end=t + duration,
            )
        )

    def fade_out(self, blade: Blade, t: float, duration: float):
        """
        Schedule a fade-out effect.

        Args:
            blade: The blade to fade out
            t: Start time (seconds)
            duration: Duration of fade (seconds)
        """
        obj_id = id(blade)
        self._effects.append(
            FadeOut(
                object_id=obj_id,
                t_start=t,
                t_end=t + duration,
            )
        )

    # =========================================================================
    # Blade -> Geometry Conversion
    # =========================================================================

    def _tracked_to_geometry(self, tracked: AnimationTrack) -> tuple[NDArray, NDArray, tuple[int, int, int] | None]:
        """
        Extract origin and spanning vectors from a tracked blade.

        If custom vectors have been set via set_vectors(), those are used.
        Otherwise, uses Observer.spanning_vectors_as_array() to factorize the blade.

        Returns:
            (origin, vectors, projection_axes) where origin is the origin point,
            vectors are the spanning vectors, and projection_axes are the axes
            to project onto (or None for 3D objects).
        """
        projection_axes = tracked.projection_axes

        # Use custom vectors if set
        if tracked.vectors is not None:
            origin = tracked.origin if tracked.origin is not None else zeros(3)
            return origin, tracked.vectors, projection_axes

        # Use Observer's spanning_vectors_as_array for factorization
        blade = tracked.blade
        dim = blade.dim

        # Determine origin dimension
        if dim <= 3:
            origin = zeros(3)
        else:
            origin = zeros(dim)

        # Get spanning vectors via Observer (which uses Blade.spanning_vectors())
        vectors = self._observer.spanning_vectors_as_array(blade)

        if vectors is None:
            # Fallback for grade 0 or unknown types
            vectors = array([zeros(dim)])

        return origin, vectors, projection_axes

    # =========================================================================
    # Session Control
    # =========================================================================

    def start(self, live: bool = False):
        """
        Start an animation session.

        Args:
            live: If True, render immediately on each capture.
                  If False (default), accumulate snapshots for later playback.
        """
        self._live = live
        self._started = True
        self._snapshots = []
        self._last_render_time = None
        self._start_wall_time = time_module.time()

        if live:
            # Initialize renderer with all tracked objects (invisible)
            for tracked in self._tracks.values():
                origin, vectors, projection_axes = self._tracked_to_geometry(tracked)
                self._renderer.add_object(
                    obj_id=tracked.obj_id,
                    grade=tracked.grade,
                    origin=origin,
                    vectors=vectors,
                    color=tracked.color,
                    opacity=0.0,
                    projection_axes=projection_axes,
                )
            self._renderer.show()
            _bring_window_to_front()

    def capture(self, t: float):
        """
        Capture the current state of all tracked objects at time t.

        In batch mode, stores snapshot for later playback.
        In live mode, renders immediately if it's time for a new frame.

        Args:
            t: Current animation time (seconds)
        """
        if not self._started:
            raise RuntimeError("Must call start() before capture()")

        # Build snapshot
        states: dict[int, tuple[NDArray, NDArray, float, tuple[int, int, int] | None]] = {}

        for tracked in self._tracks.values():
            origin, vectors, projection_axes = self._tracked_to_geometry(tracked)

            # Compute opacity from effects
            opacity = compute_opacity(self._effects, tracked.obj_id, t)
            if opacity is None:
                # No effects scheduled - default to visible
                opacity = 1.0

            states[tracked.obj_id] = (np_copy(origin), np_copy(vectors), opacity, projection_axes)

        # Include current basis labels if set
        basis_labels = getattr(self, "_basis_labels", None)
        snapshot = Snapshot(t=t, states=states, basis_labels=basis_labels)
        self._snapshots.append(snapshot)

        if self._live:
            self._render_live(snapshot)

    def _render_live(self, snapshot: Snapshot):
        """Render a snapshot immediately (live mode)."""
        # Check if enough time has passed for a new frame
        frame_duration = 1.0 / self.fps
        wall_time = time_module.time() - self._start_wall_time

        if self._last_render_time is not None:
            elapsed = wall_time - self._last_render_time
            if elapsed < frame_duration:
                # Not time for a new frame yet
                return

        # Render this frame
        for obj_id, (origin, vectors, opacity, projection_axes) in snapshot.states.items():
            tracked = self._tracks.get(obj_id)
            if tracked is None:
                continue

            self._renderer.update_object(obj_id, origin, vectors, opacity, projection_axes)

        self._renderer.render()
        self._last_render_time = wall_time

    def finish(self):
        """End an animation session (live mode)."""
        self._started = False
        if self._live:
            print("Animation complete. Close window to exit.")
            # Keep window open
            if self._renderer.plotter is not None:
                self._renderer.plotter.iren.interactor.Start()

    def play(self, loop: bool = False):
        """
        Play back recorded snapshots (batch mode).

        Args:
            loop: If True, loop the animation indefinitely
        """
        if not self._snapshots:
            print("No snapshots to play.")
            return

        # Sort snapshots by time
        self._snapshots.sort(key=lambda s: s.t)

        # Initialize renderer with all tracked objects
        for tracked in self._tracks.values():
            # Use first snapshot's geometry as initial state
            first_state = self._snapshots[0].states.get(tracked.obj_id)
            if first_state:
                origin, vectors, opacity, projection_axes = first_state
            else:
                origin, vectors, projection_axes = self._tracked_to_geometry(tracked)
                opacity = 0.0

            self._renderer.add_object(
                obj_id=tracked.obj_id,
                grade=tracked.grade,
                origin=origin,
                vectors=vectors,
                color=tracked.color,
                opacity=opacity,
                projection_axes=projection_axes,
            )

        self._renderer.show()
        _bring_window_to_front()

        # Playback loop
        t_start = self._snapshots[0].t
        t_end = self._snapshots[-1].t
        t_end - t_start

        try:
            while True:
                play_start = time_module.time()

                for snapshot in self._snapshots:
                    # Calculate target wall time for this snapshot
                    target_time = play_start + (snapshot.t - t_start)

                    # Wait until it's time
                    now = time_module.time()
                    if target_time > now:
                        time_module.sleep(target_time - now)

                    # Render this frame
                    for obj_id, (origin, vectors, opacity, projection_axes) in snapshot.states.items():
                        self._renderer.update_object(obj_id, origin, vectors, opacity, projection_axes)

                    self._renderer.render()

                if not loop:
                    break

        except KeyboardInterrupt:
            pass

        print("Animation complete. Close window to exit.")
        if self._renderer.plotter is not None:
            self._renderer.plotter.iren.interactor.Start()

    # =========================================================================
    # Saving
    # =========================================================================

    def save(self, filename: str, loop: bool = True):
        """
        Save the animation to a file.

        Supports .gif and .mp4 formats. For GIFs, the loop parameter
        controls whether the animation repeats.

        Args:
            filename: Output filename (.gif or .mp4)
            loop: If True (default), GIF will loop indefinitely
        """
        if not self._snapshots:
            print("No snapshots to save.")
            return

        # Sort snapshots by time
        self._snapshots.sort(key=lambda s: s.t)

        # Determine format
        ext = filename.lower().split(".")[-1]
        if ext not in ("gif", "mp4"):
            raise ValueError(f"Unsupported format: {ext}. Use .gif or .mp4")

        # Create off-screen renderer
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self._renderer.theme.background)
        plotter.window_size = self._renderer._size

        if self._renderer._show_basis:
            from morphis.visualization.drawing import draw_coordinate_basis

            draw_coordinate_basis(plotter, color=self._renderer.theme.axis_color)

        # Set camera if configured
        if hasattr(self, "_camera_position") and self._camera_position:
            plotter.camera.position = self._camera_position
        if hasattr(self, "_camera_focal") and self._camera_focal:
            plotter.camera.focal_point = self._camera_focal

        # Add initial objects
        actors: dict[int, tuple] = {}
        for tracked in self._tracks.values():
            first_state = self._snapshots[0].states.get(tracked.obj_id)
            if first_state:
                origin, vectors, opacity, projection_axes = first_state
            else:
                origin, vectors, projection_axes = self._tracked_to_geometry(tracked)
                opacity = 0.0

            edges_actor, faces_actor = self._add_object_to_plotter(
                plotter, tracked, origin, vectors, opacity, projection_axes
            )
            actors[tracked.obj_id] = (edges_actor, faces_actor, tracked.grade)

        # Collect frames
        frames = []
        for snapshot in self._snapshots:
            # Update all objects
            for obj_id, (origin, vectors, opacity, projection_axes) in snapshot.states.items():
                if obj_id not in actors:
                    continue

                edges_actor, faces_actor, grade = actors[obj_id]
                self._update_actor_geometry(
                    plotter, edges_actor, faces_actor, grade, origin, vectors, opacity, projection_axes
                )

            # Capture frame
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            frames.append(frame)

        plotter.close()

        # Save based on format
        if ext == "gif":
            self._save_gif(filename, frames, loop)
        else:
            self._save_mp4(filename, frames)

        print(f"Saved animation to {filename}")

    def _add_object_to_plotter(self, plotter, tracked, origin, vectors, opacity, projection_axes=None):
        """Add an object to an off-screen plotter, return actors."""
        from morphis.visualization.drawing import (
            _create_arrow_mesh,
            _create_origin_marker,
            create_bivector_mesh,
            create_quadvector_mesh,
            create_trivector_mesh,
        )

        # Helper to project vectors if needed
        def project_to_3d(vec, axes):
            if axes is None or len(vec) <= 3:
                v = vec[:3] if len(vec) >= 3 else array([*vec, *[0.0] * (3 - len(vec))])
                return v
            return array([vec[axes[0]], vec[axes[1]], vec[axes[2]]])

        if tracked.grade == 1:
            direction = vectors[0] if vectors.ndim > 1 else vectors
            direction_3d = project_to_3d(direction, projection_axes)
            origin_3d = project_to_3d(origin, projection_axes)
            edges_mesh = _create_arrow_mesh(origin_3d, direction_3d)
            origin_mesh = _create_origin_marker(origin_3d)
            edges_actor = plotter.add_mesh(edges_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            plotter.add_mesh(origin_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            faces_actor = None

        elif tracked.grade == 2:
            u, v = vectors[0], vectors[1]
            u_3d = project_to_3d(u, projection_axes)
            v_3d = project_to_3d(v, projection_axes)
            origin_3d = project_to_3d(origin, projection_axes)
            edges_mesh, faces_mesh, origin_mesh = create_bivector_mesh(origin_3d, u_3d, v_3d)
            edges_actor = plotter.add_mesh(edges_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            faces_actor = plotter.add_mesh(faces_mesh, color=tracked.color, opacity=opacity * 0.25, smooth_shading=True)
            plotter.add_mesh(origin_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)

        elif tracked.grade == 3:
            u, v, w = vectors[0], vectors[1], vectors[2]
            u_3d = project_to_3d(u, projection_axes)
            v_3d = project_to_3d(v, projection_axes)
            w_3d = project_to_3d(w, projection_axes)
            origin_3d = project_to_3d(origin, projection_axes)
            edges_mesh, faces_mesh, origin_mesh = create_trivector_mesh(origin_3d, u_3d, v_3d, w_3d)
            edges_actor = plotter.add_mesh(edges_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            faces_actor = plotter.add_mesh(faces_mesh, color=tracked.color, opacity=opacity * 0.2, smooth_shading=True)
            plotter.add_mesh(origin_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)

        elif tracked.grade == 4:
            u, v, w, x = vectors[0], vectors[1], vectors[2], vectors[3]
            axes = projection_axes or (0, 1, 2)
            edges_mesh, faces_mesh, origin_mesh = create_quadvector_mesh(origin, u, v, w, x, projection_axes=axes)
            edges_actor = plotter.add_mesh(edges_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            faces_actor = plotter.add_mesh(faces_mesh, color=tracked.color, opacity=opacity * 0.15, smooth_shading=True)
            plotter.add_mesh(origin_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)

        else:
            raise NotImplementedError(f"Grade {tracked.grade} not supported")

        return edges_actor, faces_actor

    def _update_actor_geometry(
        self, plotter, edges_actor, faces_actor, grade, origin, vectors, opacity, projection_axes=None
    ):
        """Update actor geometry and opacity."""
        from morphis.visualization.drawing import (
            _create_arrow_mesh,
            create_bivector_mesh,
            create_quadvector_mesh,
            create_trivector_mesh,
        )

        # Helper to project vectors if needed
        def project_to_3d(vec, axes):
            if axes is None or len(vec) <= 3:
                v = vec[:3] if len(vec) >= 3 else array([*vec, *[0.0] * (3 - len(vec))])
                return v
            return array([vec[axes[0]], vec[axes[1]], vec[axes[2]]])

        if grade == 1:
            direction = vectors[0] if vectors.ndim > 1 else vectors
            direction_3d = project_to_3d(direction, projection_axes)
            origin_3d = project_to_3d(origin, projection_axes)
            edges_mesh = _create_arrow_mesh(origin_3d, direction_3d)
            edges_actor.mapper.SetInputData(edges_mesh)

        elif grade == 2:
            u, v = vectors[0], vectors[1]
            u_3d = project_to_3d(u, projection_axes)
            v_3d = project_to_3d(v, projection_axes)
            origin_3d = project_to_3d(origin, projection_axes)
            edges_mesh, faces_mesh, _ = create_bivector_mesh(origin_3d, u_3d, v_3d)
            edges_actor.mapper.SetInputData(edges_mesh)
            if faces_actor is not None:
                faces_actor.mapper.SetInputData(faces_mesh)

        elif grade == 3:
            u, v, w = vectors[0], vectors[1], vectors[2]
            u_3d = project_to_3d(u, projection_axes)
            v_3d = project_to_3d(v, projection_axes)
            w_3d = project_to_3d(w, projection_axes)
            origin_3d = project_to_3d(origin, projection_axes)
            edges_mesh, faces_mesh, _ = create_trivector_mesh(origin_3d, u_3d, v_3d, w_3d)
            edges_actor.mapper.SetInputData(edges_mesh)
            if faces_actor is not None:
                faces_actor.mapper.SetInputData(faces_mesh)

        elif grade == 4:
            u, v, w, x = vectors[0], vectors[1], vectors[2], vectors[3]
            axes = projection_axes or (0, 1, 2)
            edges_mesh, faces_mesh, _ = create_quadvector_mesh(origin, u, v, w, x, projection_axes=axes)
            edges_actor.mapper.SetInputData(edges_mesh)
            if faces_actor is not None:
                faces_actor.mapper.SetInputData(faces_mesh)

        # Update opacity
        edges_actor.GetProperty().SetOpacity(opacity)
        if faces_actor is not None:
            if grade == 2:
                face_opacity = opacity * 0.25
            elif grade == 3:
                face_opacity = opacity * 0.2
            elif grade == 4:
                face_opacity = opacity * 0.15
            else:
                face_opacity = opacity * 0.2
            faces_actor.GetProperty().SetOpacity(face_opacity)

    def _save_gif(self, filename: str, frames: list, loop: bool):
        """Save frames as a GIF."""
        import imageio.v3 as iio

        # Calculate duration per frame in milliseconds
        if len(self._snapshots) > 1:
            total_time = self._snapshots[-1].t - self._snapshots[0].t
            duration_ms = int((total_time / len(frames)) * 1000)
        else:
            duration_ms = int(1000 / self.fps)

        # Ensure minimum duration
        duration_ms = max(duration_ms, 20)

        iio.imwrite(
            filename,
            frames,
            extension=".gif",
            duration=duration_ms,
            loop=0 if loop else 1,  # 0 = infinite loop
        )

    def _save_mp4(self, filename: str, frames: list):
        """Save frames as an MP4."""
        import imageio.v3 as iio

        iio.imwrite(filename, frames, fps=self.fps)

    # =========================================================================
    # Camera Control
    # =========================================================================

    def camera(self, position=None, focal_point=None):
        """Set camera position and/or focal point."""
        self._renderer.camera(position=position, focal_point=focal_point)
        # Store for save() to use
        if position is not None:
            self._camera_position = position
        if focal_point is not None:
            self._camera_focal = focal_point

    def set_basis_labels(self, labels: tuple[str, str, str]):
        """
        Set custom labels for the coordinate basis axes.

        Args:
            labels: Tuple of 3 labels for (x, y, z) axes, e.g., ("e₁", "e₂", "e₃")
        """
        self._basis_labels = labels

    def close(self):
        """Close the animation window."""
        self._renderer.close()


def _bring_window_to_front():
    """Bring the current application window to front (macOS only)."""
    if sys.platform == "darwin":
        try:
            from AppKit import NSApp, NSApplication

            NSApplication.sharedApplication()
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            pass
