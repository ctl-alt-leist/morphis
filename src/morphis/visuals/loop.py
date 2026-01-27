"""
Animation - Observer and Recorder

The Animation class observes geometric objects (blades) and records their
state over time. It is completely ignorant of what transformations are
applied to the objects - it just reads their current state when asked.

Key concepts:
- watch(blade): Register a blade to observe
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
from typing import Any

from numpy import array, copy as np_copy, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.elements.frame import Frame
from morphis.elements.vector import Vector
from morphis.utils.observer import Observer
from morphis.visuals.effects import Effect, FadeIn, FadeOut, compute_opacity
from morphis.visuals.renderer import Renderer
from morphis.visuals.theme import Color, Theme


class Snapshot(BaseModel):
    """State of all tracked objects at a specific time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: float
    # obj_id -> (origin, vectors, opacity, projection_axes)
    states: dict[int, tuple[Any, Any, float, tuple[int, int, int] | None]]
    basis_labels: tuple[str, str, str] | None = None  # Optional basis labels for this frame


class AnimationTrack(BaseModel):
    """Animation-specific tracking info for a blade or frame."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: Any  # Vector | Frame - using Any for compatibility
    obj_id: int
    color: Color
    grade: int  # For blades; -1 for frames
    is_frame: bool = False  # True if target is a Frame
    filled: bool = False  # For frames: whether to show edges and faces
    vectors: Any | None = None  # NDArray | None - Override for spanning vectors
    origin: Any | None = None  # NDArray | None - Override for origin


class Animation:
    """
    Animation observer and recorder.

    Observes blades and records their state over time. Supports both batch
    mode (record all, then play) and live mode (render as you go).

    Example (batch mode):
        anim = Animation(frame_rate=60)
        anim.watch(q, color=(0.85, 0.2, 0.2))
        anim.fade_in(q, t=0.0, duration=1.0)

        anim.start()
        for t in times:
            q.data[...] = transform(q)
            anim.capture(t)
        anim.play()

    Example (live mode):
        anim = Animation(frame_rate=60)
        anim.watch(q)

        anim.start(live=True)
        for t in times:
            q.data[...] = transform(q)
            anim.capture(t)
        anim.finish()
    """

    def __init__(
        self,
        frame_rate: int = 60,
        theme: str | Theme = "obsidian",
        size: tuple[int, int] = (1800, 1350),
        show_basis: bool = True,
        auto_camera: bool = True,
        fps: int | None = None,  # Deprecated alias for frame_rate
    ):
        # Handle deprecated fps parameter
        if fps is not None:
            frame_rate = fps
        self.frame_rate = frame_rate
        self._auto_camera = auto_camera
        self._projection_axes: tuple[int, int, int] = (0, 1, 2)  # Default projection
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

    def watch(self, *targets: Vector | Frame, color: Color | None = None, filled: bool = False) -> int | list[int]:
        """
        Register one or more blades or frames to observe.

        Args:
            *targets: Vectors or Frames to watch
            color: Optional color override (applies to all)
            filled: For frames, whether to show edges and faces of spanned shape

        Returns:
            Object ID(s) for the target(s)
        """
        ids = []
        for target in targets:
            obj_id = id(target)

            if obj_id in self._tracks:
                ids.append(obj_id)
                continue

            target_color = color if color is not None else self._renderer._next_color()

            is_frame = isinstance(target, Frame)

            if is_frame:
                # Frame: grade=-1 signals frame rendering
                self._tracks[obj_id] = AnimationTrack(
                    target=target,
                    obj_id=obj_id,
                    color=target_color,
                    grade=-1,
                    is_frame=True,
                    filled=filled,
                )
            else:
                # Vector: watch in Observer for core functionality
                self._observer.watch(target)
                self._tracks[obj_id] = AnimationTrack(
                    target=target,
                    obj_id=obj_id,
                    color=target_color,
                    grade=target.grade,
                    is_frame=False,
                )

            ids.append(obj_id)

        return ids[0] if len(ids) == 1 else ids

    # Alias for backward compatibility
    track = watch

    def unwatch(self, *targets: Vector | Frame):
        """Stop watching one or more blades or frames."""
        for target in targets:
            obj_id = id(target)
            if obj_id in self._tracks:
                track = self._tracks[obj_id]
                del self._tracks[obj_id]
                if not track.is_frame:
                    self._observer.unwatch(target)
                self._renderer.remove_object(obj_id)

    # Alias for backward compatibility
    untrack = unwatch

    def set_vectors(self, blade: Vector, vectors: NDArray, origin: NDArray | None = None):
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

    def set_projection(
        self,
        axes: tuple[int, int, int],
        labels: tuple[str, str, str] | None = None,
    ):
        """
        Set the coordinate projection for the canvas.

        This determines which 3 coordinate axes are displayed. All watched
        objects are then projected onto this coordinate system based on
        their dimensionality.

        Automatically generates basis labels from axes unless overridden.

        Examples:
            anim.set_projection((0, 1, 2))  # Show e1, e2, e3
            anim.set_projection((1, 2, 3))  # Show e2, e3, e4
            anim.set_projection((0, 1, 2), labels=("x", "y", "z"))  # Custom

        Args:
            axes: Tuple of 3 axis indices (0-indexed)
            labels: Optional custom labels (auto-generated if not provided)
        """
        self._projection_axes = axes

        # Auto-generate basis labels from axes (0-indexed to 1-indexed)
        if labels is None:
            labels = tuple(f"$\\mathbf{{e}}_{i + 1}$" for i in axes)
        self._basis_labels = labels

        # If in live mode, update renderer immediately
        if self._live and self._started:
            self._renderer.set_basis_labels(labels)

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

    def __contains__(self, blade: Vector) -> bool:
        """Check if a blade is being tracked."""
        return id(blade) in self._tracks

    def __iter__(self):
        """Iterate over tracked blades."""
        return iter(track.target for track in self._tracks.values())

    # =========================================================================
    # Effects
    # =========================================================================

    def fade_in(self, target: Vector | Frame, t: float, duration: float):
        """
        Schedule a fade-in effect.

        Args:
            target: The blade or frame to fade in
            t: Start time (seconds)
            duration: Duration of fade (seconds)
        """
        obj_id = id(target)
        self._effects.append(
            FadeIn(
                object_id=obj_id,
                t_start=t,
                t_end=t + duration,
            )
        )

    def fade_out(self, target: Vector | Frame, t: float, duration: float):
        """
        Schedule a fade-out effect.

        Args:
            target: The blade or frame to fade out
            t: Start time (seconds)
            duration: Duration of fade (seconds)
        """
        obj_id = id(target)
        self._effects.append(
            FadeOut(
                object_id=obj_id,
                t_start=t,
                t_end=t + duration,
            )
        )

    # =========================================================================
    # Vector/Frame -> Geometry Conversion
    # =========================================================================

    def _tracked_to_geometry(self, tracked: AnimationTrack) -> tuple[NDArray, NDArray, tuple[int, int, int] | None]:
        """
        Extract origin and spanning vectors from a tracked blade or frame.

        If custom vectors have been set via set_vectors(), those are used.
        For frames, vectors are extracted directly from the Frame object.
        For blades, uses Observer.spanning_vectors_as_array() to factorize.

        Returns:
            (origin, vectors, projection_axes) where origin is the origin point,
            vectors are the spanning vectors, and projection_axes are the axes
            to project onto.
        """
        projection_axes = self._projection_axes

        # Use custom vectors if set
        if tracked.vectors is not None:
            origin = tracked.origin if tracked.origin is not None else zeros(3)
            return origin, tracked.vectors, projection_axes

        if tracked.is_frame:
            # Frame: extract vectors directly
            frame = tracked.target
            dim = frame.dim

            # Determine origin dimension
            if dim <= 3:
                origin = tracked.origin if tracked.origin is not None else zeros(3)
            else:
                origin = tracked.origin if tracked.origin is not None else zeros(dim)

            # Get vectors from frame: shape (k, d) -> list of k vectors
            vectors = frame.data.copy()
            return origin, vectors, projection_axes

        # Vector: use Observer's spanning_vectors_as_array for factorization
        blade = tracked.target
        dim = blade.dim

        # Determine origin dimension
        if dim <= 3:
            origin = zeros(3)
        else:
            origin = zeros(dim)

        # Get spanning vectors via Observer (which uses Vector.spanning_vectors())
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
                    filled=tracked.filled,
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
        frame_duration = 1.0 / self.frame_rate
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
                filled=tracked.filled,
            )

        # Apply auto camera if enabled and no manual camera set
        if self._auto_camera and not hasattr(self, "_camera_position"):
            self._compute_dynamic_camera()
            self._renderer.camera(position=self._camera_position, focal_point=self._camera_focal)

        # Set initial basis labels from first snapshot
        current_labels = None
        if self._snapshots[0].basis_labels:
            current_labels = self._snapshots[0].basis_labels
            self._renderer.set_basis_labels(current_labels)

        self._renderer.show()
        _bring_window_to_front()

        # Playback loop
        t_start = self._snapshots[0].t

        try:
            while True:
                play_start = time_module.time()

                for snapshot in self._snapshots:
                    # Calculate target wall time for this snapshot
                    target_time = play_start + (snapshot.t - t_start)

                    # Wait until it's time, but keep processing window events
                    while time_module.time() < target_time:
                        if self._renderer.plotter is not None:
                            self._renderer.plotter.iren.process_events()
                        time_module.sleep(0.001)  # Small sleep to avoid busy-waiting

                    # Update basis labels if changed
                    if snapshot.basis_labels != current_labels:
                        if snapshot.basis_labels is not None:
                            self._renderer.set_basis_labels(snapshot.basis_labels)
                        current_labels = snapshot.basis_labels

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

        # Track basis label actors for updating during animation
        basis_label_actors = None
        current_labels = None

        if self._renderer._show_basis:
            from morphis.visuals.drawing.vectors import draw_coordinate_basis

            # Use first snapshot's labels if available
            initial_labels = self._snapshots[0].basis_labels if self._snapshots else None
            current_labels = initial_labels
            basis_label_actors = draw_coordinate_basis(
                plotter, color=self._renderer.theme.axis_color, labels=initial_labels
            )

        # Apply auto camera if enabled and no manual camera set
        if self._auto_camera and not hasattr(self, "_camera_position"):
            self._compute_dynamic_camera()

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
            actors[tracked.obj_id] = (edges_actor, faces_actor, tracked.grade, tracked.filled)

        # Collect frames
        frames = []
        for snapshot in self._snapshots:
            # Update basis labels if changed
            if self._renderer._show_basis and snapshot.basis_labels != current_labels:
                if snapshot.basis_labels is not None:
                    # Remove old labels
                    if basis_label_actors:
                        for actor in basis_label_actors:
                            plotter.remove_actor(actor)
                    # Draw new labels
                    basis_label_actors = self._draw_basis_labels_to_plotter(plotter, snapshot.basis_labels)
                current_labels = snapshot.basis_labels

            # Update all objects
            for obj_id, (origin, vectors, opacity, projection_axes) in snapshot.states.items():
                if obj_id not in actors:
                    continue

                edges_actor, faces_actor, grade, filled = actors[obj_id]
                self._update_actor_geometry(
                    plotter, edges_actor, faces_actor, grade, origin, vectors, opacity, projection_axes, filled
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

    def _draw_basis_labels_to_plotter(self, plotter, labels: tuple[str, str, str]) -> list:
        """Draw only basis labels to a plotter (not arrows), return actors."""
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
            actor = plotter.add_point_labels(
                [label_pos],
                [axis_name],
                font_size=12,
                text_color=self._renderer.theme.axis_color,
                point_size=0,
                shape=None,
                show_points=False,
                always_visible=True,
            )
            label_actors.append(actor)
        return label_actors

    def _add_object_to_plotter(self, plotter, tracked, origin, vectors, opacity, projection_axes=None):
        """Add an object to an off-screen plotter, return actors."""
        from morphis.visuals.drawing.vectors import (
            _create_arrow_mesh,
            _create_origin_marker,
            create_bivector_mesh,
            create_frame_mesh,
            create_quadvector_mesh,
            create_trivector_mesh,
        )

        # Helper to project vectors if needed
        def project_to_3d(blade, axes):
            if axes is None or len(blade) <= 3:
                v = blade[:3] if len(blade) >= 3 else array([*blade, *[0.0] * (3 - len(blade))])
                return v
            return array([blade[axes[0]], blade[axes[1]], blade[axes[2]]])

        if tracked.grade == -1:
            # Frame: k arrows from origin (optionally with edges/faces)
            edges_mesh, faces_mesh, origin_mesh = create_frame_mesh(
                origin, vectors, projection_axes=projection_axes, filled=tracked.filled
            )
            edges_actor = plotter.add_mesh(edges_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            plotter.add_mesh(origin_mesh, color=tracked.color, opacity=opacity, smooth_shading=True)
            if faces_mesh is not None:
                faces_actor = plotter.add_mesh(
                    faces_mesh, color=tracked.color, opacity=opacity * 0.2, smooth_shading=True
                )
            else:
                faces_actor = None

        elif tracked.grade == 1:
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
        self, plotter, edges_actor, faces_actor, grade, origin, vectors, opacity, projection_axes=None, filled=False
    ):
        """Update actor geometry and opacity."""
        from morphis.visuals.drawing.vectors import (
            _create_arrow_mesh,
            create_bivector_mesh,
            create_frame_mesh,
            create_quadvector_mesh,
            create_trivector_mesh,
        )

        # Helper to project vectors if needed
        def project_to_3d(blade, axes):
            if axes is None or len(blade) <= 3:
                v = blade[:3] if len(blade) >= 3 else array([*blade, *[0.0] * (3 - len(blade))])
                return v
            return array([blade[axes[0]], blade[axes[1]], blade[axes[2]]])

        if grade == -1:
            # Frame: k arrows from origin (optionally with edges/faces)
            edges_mesh, faces_mesh, _ = create_frame_mesh(
                origin, vectors, projection_axes=projection_axes, filled=filled
            )
            edges_actor.mapper.SetInputData(edges_mesh)
            if faces_actor is not None and faces_mesh is not None:
                faces_actor.mapper.SetInputData(faces_mesh)

        elif grade == 1:
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
            duration_ms = int(1000 / self.frame_rate)

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

        iio.imwrite(filename, frames, fps=self.frame_rate)

    # =========================================================================
    # Camera Control
    # =========================================================================

    def camera(self, position=None, focal_point=None):
        """
        Set camera position and/or focal point.

        Calling this method disables auto_camera positioning.
        """
        self._auto_camera = False  # Manual camera takes precedence
        self._renderer.camera(position=position, focal_point=focal_point)
        # Store for save() to use
        if position is not None:
            self._camera_position = position
        if focal_point is not None:
            self._camera_focal = focal_point

    def _compute_dynamic_camera(self):
        """
        Compute reasonable camera position based on watched objects' extent.

        Strategy:
        - Compute bounding box of all tracked objects across all snapshots
        - Position camera to see entire scene with margin
        - Default viewing angle: isometric-ish (positive x, negative y, positive z)
        """
        from numpy import array, max as np_max, min as np_min, zeros
        from numpy.linalg import norm

        if not self._snapshots:
            # Default position if no snapshots
            self._camera_position = (4.0, -3.0, 3.5)
            self._camera_focal = (0.0, 0.0, 0.0)
            return

        # Collect all vertex positions across all snapshots
        all_points = []

        for snapshot in self._snapshots:
            for _obj_id, (origin, vectors, _opacity, proj_axes) in snapshot.states.items():
                # Project to 3D if needed
                def to_3d(v, axes):
                    if axes is None or len(v) <= 3:
                        return v[:3] if len(v) >= 3 else array([*v, *zeros(3 - len(v))])
                    return array([v[axes[0]], v[axes[1]], v[axes[2]]])

                origin_3d = to_3d(origin, proj_axes)
                all_points.append(origin_3d)

                # Add extent from vectors
                if vectors.ndim == 2:
                    for i in range(vectors.shape[0]):
                        vec_3d = to_3d(vectors[i], proj_axes)
                        all_points.append(origin_3d + vec_3d)

        if not all_points:
            self._camera_position = (4.0, -3.0, 3.5)
            self._camera_focal = (0.0, 0.0, 0.0)
            return

        points = array(all_points)

        # Bounding box
        min_pt = np_min(points, axis=0)
        max_pt = np_max(points, axis=0)
        center = (min_pt + max_pt) / 2
        extent = np_max(max_pt - min_pt)

        # Camera distance based on extent (with margin)
        distance = extent * 4

        # Isometric-like viewing angle: (1, -0.6, 0.8) normalized
        direction = array([1.0, -0.6, 0.8])
        direction = direction / norm(direction)

        self._camera_position = tuple(center + direction * distance)
        self._camera_focal = tuple(center)

    def set_basis_labels(self, labels: tuple[str, str, str]):
        """
        Set custom labels for the coordinate basis axes.

        Args:
            labels: Tuple of 3 labels for (x, y, z) axes, e.g., ("e1", "e2", "e3")
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
