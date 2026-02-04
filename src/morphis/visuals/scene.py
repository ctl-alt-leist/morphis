"""
Scene - Unified Visualization Interface

Scene provides a single interface for both static and animated visualization,
replacing the separate Canvas and Animation classes. It uses a pluggable
backend system for rendering.
"""

from __future__ import annotations

import sys
import time as time_module
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from numpy import array, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.elements.base import Element
from morphis.elements.frame import Frame
from morphis.elements.vector import Vector
from morphis.visuals.backends import get_backend
from morphis.visuals.theme import Color, Theme, get_theme


if TYPE_CHECKING:
    from morphis.visuals.backends.protocol import RenderBackend


class SceneEffect(BaseModel):
    """Effect wrapper that uses string element IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_id: str
    t_start: float
    t_end: float
    effect_type: str  # "fade_in" or "fade_out"

    def evaluate(self, t: float) -> float:
        """Evaluate opacity at time t."""
        if t <= self.t_start:
            return 0.0 if self.effect_type == "fade_in" else 1.0
        if t >= self.t_end:
            return 1.0 if self.effect_type == "fade_in" else 0.0

        progress = (t - self.t_start) / (self.t_end - self.t_start)
        if self.effect_type == "fade_in":
            return progress
        else:
            return 1.0 - progress

    def is_active(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end


class TrackedElement(BaseModel):
    """Internal tracking for elements added to the scene."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_id: str
    element: Any  # Element instance
    backend_ids: list[str]  # IDs from backend for actors
    color: Color
    opacity: float
    representation: str
    extra: dict  # Additional settings


class Snapshot(BaseModel):
    """State of all tracked elements at a specific time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: float
    states: dict[str, dict]  # element_id -> state dict


class Scene:
    """
    Unified visualization interface for static and animated scenes.

    Scene manages elements with visual representations, supporting both
    interactive viewing and animation recording/playback.

    Static example:
        scene = Scene(theme="obsidian")
        scene.add(v, representation="arrow", color=RED)
        scene.show()

    Animation example:
        scene = Scene(theme="obsidian", frame_rate=30)
        scene.add(F, representation="arrows", color=RED)
        scene.capture(0.0)
        for t in times:
            F.data[...] = transform(F)
            scene.capture(t)
        scene.play()
    """

    def __init__(
        self,
        projection: tuple[int, ...] | None = None,
        theme: str | Theme = "obsidian",
        size: tuple[int, int] = (1200, 900),
        frame_rate: int = 30,
        backend: str = "pyvista",
        show_basis: bool = True,
    ):
        """
        Initialize the scene.

        Args:
            projection: Axes to project nD -> 3D (default: (0, 1, 2))
            theme: Visual theme name or Theme instance
            size: Window size (width, height)
            frame_rate: Frames per second for animation
            backend: Rendering backend name
            show_basis: Show coordinate basis arrows
        """
        if isinstance(theme, str):
            theme = get_theme(theme)

        self._theme = theme
        self._size = size
        self._frame_rate = frame_rate
        self._projection = projection or (0, 1, 2)
        self._show_basis = show_basis

        # Get backend but don't initialize yet (lazy)
        self._backend: RenderBackend = get_backend(backend)
        self._backend_initialized = False

        # Element tracking
        self._elements: dict[str, TrackedElement] = {}
        self._color_index = 0

        # Animation state
        self._snapshots: list[Snapshot] = []
        self._recording = False
        self._effects: list[SceneEffect] = []

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def theme(self) -> Theme:
        """Current theme."""
        return self._theme

    @property
    def frame_rate(self) -> int:
        """Animation frame rate."""
        return self._frame_rate

    # =========================================================================
    # Backend Management
    # =========================================================================

    def _ensure_backend(self):
        """Initialize backend if needed."""
        if not self._backend_initialized:
            self._backend.initialize(
                size=self._size,
                theme=self._theme,
                show_basis=self._show_basis,
            )
            self._backend_initialized = True

    def _next_color(self) -> Color:
        """Get next color from palette."""
        color = self._theme.palette[self._color_index % len(self._theme.palette)]
        self._color_index += 1
        return color

    # =========================================================================
    # Element Management
    # =========================================================================

    def add(
        self,
        element: Element,
        representation: str | None = None,
        color: Color | None = None,
        opacity: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Add an element to the scene.

        Args:
            element: The element to add (Vector, Frame, Surface, etc.)
            representation: Visual representation type
            color: RGB color tuple (uses palette if None)
            opacity: Opacity [0, 1]
            **kwargs: Additional representation-specific options

        Returns:
            Element ID for later reference
        """
        self._ensure_backend()

        element_id = str(uuid4())
        color = color if color is not None else self._next_color()

        # Determine default representation based on element type
        if representation is None:
            representation = self._default_representation(element)

        # Create backend visuals
        backend_ids = self._create_visuals(element, representation, color, opacity, kwargs)

        self._elements[element_id] = TrackedElement(
            element_id=element_id,
            element=element,
            backend_ids=backend_ids,
            color=color,
            opacity=opacity,
            representation=representation,
            extra=kwargs,
        )

        return element_id

    def remove(self, element_id: str) -> None:
        """Remove an element from the scene."""
        if element_id not in self._elements:
            return

        tracked = self._elements.pop(element_id)

        for bid in tracked.backend_ids:
            self._backend.remove(bid)

    def _default_representation(self, element: Element) -> str:
        """Determine default representation for an element."""
        from morphis.elements.surface import Surface

        if isinstance(element, Surface):
            return "mesh"
        elif isinstance(element, Frame):
            return "arrows"
        elif isinstance(element, Vector):
            if element.grade == 1:
                if element.lot and element.lot != ():
                    return "arrows"
                return "arrow"
            elif element.grade == 2:
                return "span"
            else:
                return "span"
        return "points"

    def _create_visuals(
        self,
        element: Element,
        representation: str,
        color: Color,
        opacity: float,
        kwargs: dict,
    ) -> list[str]:
        """Create backend visuals for an element."""
        from morphis.elements.surface import Surface

        backend_ids = []

        if isinstance(element, Surface):
            # Surface -> mesh
            bid = self._backend.add_mesh(
                element.vertices.data,
                element.faces,
                color=color,
                opacity=opacity,
                smooth_shading=kwargs.get("smooth_shading", True),
                show_edges=kwargs.get("show_edges", False),
            )
            backend_ids.append(bid)

        elif isinstance(element, Frame):
            # Frame -> arrows from origin
            origin = kwargs.get("origin", zeros(3))
            vectors = element.data

            # Project to 3D if needed
            origin_3d = self._project_point(origin)
            vecs_3d = self._project_vectors(vectors)

            origins = array([origin_3d] * len(vecs_3d))
            bid = self._backend.add_arrows(
                origins,
                vecs_3d,
                color=color,
                opacity=opacity,
                tip_length=kwargs.get("tip_length", 0.12),
                shaft_radius=kwargs.get("shaft_radius", 0.008),
            )
            backend_ids.append(bid)

            # Optional filled visualization
            if kwargs.get("filled", False) and len(vecs_3d) >= 2:
                bid = self._backend.add_span(
                    origin_3d,
                    vecs_3d[:3],  # Max 3 vectors for span
                    color=color,
                    opacity=opacity * 0.3,
                    filled=True,
                )
                backend_ids.append(bid)

        elif isinstance(element, Vector):
            if element.grade == 1:
                data = element.data
                if element.lot and element.lot != ():
                    # Collection of vectors
                    origins = kwargs.get("origins", zeros((len(data), 3)))
                    vecs_3d = self._project_vectors(data)
                    origins_3d = self._project_vectors(origins)

                    bid = self._backend.add_arrows(
                        origins_3d,
                        vecs_3d,
                        color=color,
                        opacity=opacity,
                    )
                    backend_ids.append(bid)
                else:
                    # Single vector
                    origin = kwargs.get("origin", zeros(3))
                    origin_3d = self._project_point(origin)
                    vec_3d = self._project_point(data)

                    bid = self._backend.add_arrows(
                        origin_3d,
                        vec_3d,
                        color=color,
                        opacity=opacity,
                    )
                    backend_ids.append(bid)

            elif element.grade >= 2:
                # Bivector/trivector as span
                from morphis.operations.factorization import spanning_vectors

                factors = spanning_vectors(element)
                origin = kwargs.get("origin", zeros(3))
                origin_3d = self._project_point(origin)
                vecs = array([self._project_point(f.data) for f in factors])

                bid = self._backend.add_span(
                    origin_3d,
                    vecs,
                    color=color,
                    opacity=opacity * 0.3 if element.grade == 2 else opacity * 0.2,
                    filled=kwargs.get("filled", True),
                )
                backend_ids.append(bid)

        return backend_ids

    def _project_point(self, point: NDArray) -> NDArray:
        """Project a point to 3D using current projection axes."""
        point = array(point, dtype=float)
        if len(point) <= 3:
            result = zeros(3)
            result[: len(point)] = point
            return result

        ax = self._projection
        return array([point[ax[0]], point[ax[1]], point[ax[2]]])

    def _project_vectors(self, vectors: NDArray) -> NDArray:
        """Project multiple vectors to 3D."""
        vectors = array(vectors, dtype=float)
        if vectors.ndim == 1:
            return self._project_point(vectors).reshape(1, 3)

        return array([self._project_point(v) for v in vectors])

    # =========================================================================
    # Projection
    # =========================================================================

    def set_projection(self, axes: tuple[int, ...]) -> None:
        """
        Set projection axes for nD -> 3D.

        Args:
            axes: Tuple of 3 axis indices (0-indexed)
        """
        if len(axes) != 3:
            raise ValueError("Projection requires exactly 3 axes")

        self._projection = axes

        # Update basis labels
        labels = tuple(f"$\\mathbf{{e}}_{i + 1}$" for i in axes)
        if self._backend_initialized:
            self._backend.set_basis_labels(labels)

    # =========================================================================
    # Camera
    # =========================================================================

    def camera(
        self,
        position: tuple[float, float, float] | None = None,
        focal_point: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
    ) -> None:
        """Set camera position and orientation."""
        self._ensure_backend()
        self._backend.set_camera(position=position, focal_point=focal_point, up=up)

    def reset_camera(self) -> None:
        """Reset camera to fit all objects."""
        self._ensure_backend()
        self._backend.reset_camera()

    def set_clipping_range(self, near: float, far: float) -> None:
        """
        Set camera clipping range.

        Use this to prevent objects from being clipped when they move
        far from the camera.

        Args:
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self._ensure_backend()
        self._backend.set_clipping_range(near, far)

    # =========================================================================
    # Lighting
    # =========================================================================

    def add_light(
        self,
        position: tuple[float, float, float] = (1, 1, 1),
        focal_point: tuple[float, float, float] = (0, 0, 0),
        intensity: float = 1.0,
        color: Color | None = None,
        directional: bool = True,
        attenuation: tuple[float, float, float] | None = None,
    ) -> str:
        """
        Add a light to the scene.

        Args:
            position: Light position. For directional lights, this is the
                     direction the light comes FROM (normalized internally).
            focal_point: Point the light aims at (default: origin)
            intensity: Light brightness [0, inf). Default 1.0.
            color: Light color RGB. Default: white (1, 1, 1)
            directional: If True (default), parallel rays like sunlight from
                        infinity. If False, point light at position.
            attenuation: For point lights only: (constant, linear, quadratic).
                        Intensity falls as 1/(c + l*d + q*d²).
                        Examples:
                          None or (1,0,0) = no falloff (constant)
                          (0,0,1) = inverse square law
                          (1,0,0.1) = gentle falloff

        Returns:
            Light ID for later removal

        Examples:
            # Sunlight from upper-right
            scene.add_light(position=(10, -5, 8), directional=True)

            # Point light with inverse square falloff
            scene.add_light(
                position=(2, 0, 3),
                directional=False,
                attenuation=(0, 0, 1),
            )

            # Soft fill light (no falloff)
            scene.add_light(
                position=(-5, 5, 2),
                intensity=0.3,
                directional=True,
            )
        """
        self._ensure_backend()

        if color is None:
            color = (1.0, 1.0, 1.0)

        return self._backend.add_light(
            position=position,
            focal_point=focal_point,
            intensity=intensity,
            color=color,
            directional=directional,
            attenuation=attenuation,
        )

    def remove_light(self, light_id: str) -> None:
        """Remove a light from the scene."""
        self._backend.remove_light(light_id)

    def clear_lights(self) -> None:
        """Remove all user-added lights."""
        self._backend.clear_lights()

    # =========================================================================
    # Effects
    # =========================================================================

    def fade_in(
        self,
        element: Element,
        t: float,
        duration: float,
    ) -> None:
        """
        Schedule a fade-in effect for an element.

        Args:
            element: Element to fade in (must be already added)
            t: Start time in seconds
            duration: Duration of fade in seconds
        """
        element_id = self._find_element_id(element)
        if element_id is None:
            raise ValueError("Element not found in scene. Add it first.")

        self._effects.append(
            SceneEffect(
                element_id=element_id,
                t_start=t,
                t_end=t + duration,
                effect_type="fade_in",
            )
        )

    def fade_out(
        self,
        element: Element,
        t: float,
        duration: float,
    ) -> None:
        """
        Schedule a fade-out effect for an element.

        Args:
            element: Element to fade out (must be already added)
            t: Start time in seconds
            duration: Duration of fade in seconds
        """
        element_id = self._find_element_id(element)
        if element_id is None:
            raise ValueError("Element not found in scene. Add it first.")

        self._effects.append(
            SceneEffect(
                element_id=element_id,
                t_start=t,
                t_end=t + duration,
                effect_type="fade_out",
            )
        )

    def _find_element_id(self, element: Element) -> str | None:
        """Find the element ID for a given element."""
        for eid, tracked in self._elements.items():
            if tracked.element is element:
                return eid
        return None

    def _compute_opacity(self, element_id: str, t: float) -> float:
        """Compute effective opacity for an element at time t."""
        relevant = [e for e in self._effects if e.element_id == element_id]

        if not relevant:
            return 1.0  # Default visible

        # Find active effects
        active = [e for e in relevant if e.is_active(t)]

        if not active:
            # Check if past all effects
            past = [e for e in relevant if t > e.t_end]
            if past:
                latest = max(past, key=lambda e: e.t_end)
                return latest.evaluate(latest.t_end)
            # Before any effects
            return 0.0

        # Use most recently started active effect
        current = max(active, key=lambda e: e.t_start)
        return current.evaluate(t)

    # =========================================================================
    # Animation
    # =========================================================================

    def capture(self, t: float) -> None:
        """
        Capture current state at time t for animation.

        Args:
            t: Animation time in seconds
        """
        self._ensure_backend()
        self._recording = True

        # Capture state of all elements (including computed opacity)
        states = {}
        for element_id, tracked in self._elements.items():
            state = self._capture_element_state(tracked)
            state["opacity"] = self._compute_opacity(element_id, t)
            states[element_id] = state

        self._snapshots.append(Snapshot(t=t, states=states))

        # Sync visuals with current element state and opacity
        self._sync_visuals(t)

    def _capture_element_state(self, tracked: TrackedElement) -> dict:
        """Capture current state of an element."""
        element = tracked.element

        if hasattr(element, "vertices"):
            # Surface
            return {"vertices": element.vertices.data.copy()}
        elif isinstance(element, Frame):
            return {"data": element.data.copy()}
        elif isinstance(element, Vector):
            return {"data": element.data.copy()}

        return {}

    def _sync_visuals(self, t: float | None = None) -> None:
        """Synchronize backend visuals with current element state."""
        from morphis.elements.surface import Surface

        for element_id, tracked in self._elements.items():
            element = tracked.element

            # Compute opacity if we have time
            if t is not None:
                opacity = self._compute_opacity(element_id, t)
                for bid in tracked.backend_ids:
                    self._backend.set_opacity(bid, opacity * tracked.opacity)

            if isinstance(element, Surface):
                # Update mesh vertices
                if tracked.backend_ids:
                    self._backend.update_mesh(
                        tracked.backend_ids[0],
                        element.vertices.data,
                    )

            elif isinstance(element, Frame):
                # Update arrows
                origin = tracked.extra.get("origin", zeros(3))
                origin_3d = self._project_point(origin)
                vecs_3d = self._project_vectors(element.data)
                origins = array([origin_3d] * len(vecs_3d))

                if tracked.backend_ids:
                    self._backend.update_arrows(
                        tracked.backend_ids[0],
                        origins,
                        vecs_3d,
                    )

            elif isinstance(element, Vector) and element.grade == 1:
                # Update vector arrows
                if element.lot and element.lot != ():
                    origins = tracked.extra.get("origins", zeros((len(element.data), 3)))
                    vecs_3d = self._project_vectors(element.data)
                    origins_3d = self._project_vectors(origins)

                    if tracked.backend_ids:
                        self._backend.update_arrows(
                            tracked.backend_ids[0],
                            origins_3d,
                            vecs_3d,
                        )
                else:
                    origin = tracked.extra.get("origin", zeros(3))
                    origin_3d = self._project_point(origin)
                    vec_3d = self._project_point(element.data)

                    if tracked.backend_ids:
                        self._backend.update_arrows(
                            tracked.backend_ids[0],
                            origin_3d.reshape(1, 3),
                            vec_3d.reshape(1, 3),
                        )

        self._backend.render()

    def play(self, loop: bool = False) -> None:
        """
        Play back recorded animation.

        Args:
            loop: If True, loop indefinitely
        """
        if not self._snapshots:
            print("No snapshots to play.")
            return

        self._ensure_backend()

        # Sort by time
        self._snapshots.sort(key=lambda s: s.t)

        # Show window
        self._backend.show(interactive=False)
        _bring_window_to_front()

        t_start = self._snapshots[0].t

        try:
            while True:
                play_start = time_module.time()

                for snapshot in self._snapshots:
                    target_time = play_start + (snapshot.t - t_start)

                    # Wait for target time
                    while time_module.time() < target_time:
                        time_module.sleep(0.001)

                    # Apply snapshot state to elements
                    for element_id, state in snapshot.states.items():
                        if element_id not in self._elements:
                            continue

                        tracked = self._elements[element_id]
                        element = tracked.element

                        if "vertices" in state:
                            element.vertices.data[:] = state["vertices"]
                        elif "data" in state:
                            element.data[:] = state["data"]

                        # Apply opacity from snapshot
                        if "opacity" in state:
                            for bid in tracked.backend_ids:
                                self._backend.set_opacity(bid, state["opacity"] * tracked.opacity)

                    # Sync visuals (without time - opacity already applied)
                    self._sync_visuals()

                if not loop:
                    break

        except KeyboardInterrupt:
            pass

        print("Animation complete. Close window to exit.")
        if hasattr(self._backend, "plotter") and self._backend.plotter is not None:
            self._backend.plotter.iren.interactor.Start()

    # =========================================================================
    # Export
    # =========================================================================

    def export(self, path: str, format: str | None = None) -> None:
        """
        Export animation to file.

        Args:
            path: Output file path
            format: Format ("gif" or "mp4"). Inferred from extension if None.
        """
        if not self._snapshots:
            print("No snapshots to export.")
            return

        if format is None:
            format = path.split(".")[-1].lower()

        if format not in ("gif", "mp4"):
            raise ValueError(f"Unsupported format: {format}")

        self._snapshots.sort(key=lambda s: s.t)

        # Create off-screen plotter for export
        from pyvista import Plotter

        from morphis.visuals.drawing.vectors import draw_coordinate_basis

        plotter = Plotter(off_screen=True)
        plotter.set_background(self._theme.background)
        plotter.window_size = self._size

        # Draw coordinate basis if enabled
        if self._show_basis:
            draw_coordinate_basis(plotter, color=self._theme.axis_color)

        # Compute auto camera position based on geometry extent
        camera_pos, camera_focal = self._compute_auto_camera()
        if camera_pos is not None:
            plotter.camera.position = camera_pos
            plotter.camera.focal_point = camera_focal

        # Add all tracked elements with initial state
        element_actors: dict[str, dict] = {}  # element_id -> {actors, tracked}

        first_snapshot = self._snapshots[0]
        for element_id, tracked in self._elements.items():
            state = first_snapshot.states.get(element_id, {})
            opacity = state.get("opacity", 1.0) * tracked.opacity

            actors = self._add_element_to_plotter(plotter, tracked, opacity)
            element_actors[element_id] = {"actors": actors, "tracked": tracked}

        # Capture frames
        frames = []
        for snapshot in self._snapshots:
            # Update each element's state and opacity
            for element_id, state in snapshot.states.items():
                if element_id not in element_actors:
                    continue

                tracked = self._elements[element_id]
                element = tracked.element
                actors = element_actors[element_id]["actors"]

                # Apply data to element
                if "vertices" in state:
                    element.vertices.data[:] = state["vertices"]
                elif "data" in state:
                    element.data[:] = state["data"]

                # Compute opacity
                opacity = state.get("opacity", 1.0) * tracked.opacity

                # Update geometry and opacity
                self._update_element_in_plotter(plotter, tracked, actors, opacity)

            # Render and capture
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            frames.append(frame)

        plotter.close()

        # Save
        import imageio.v3 as iio

        if format == "gif":
            if len(self._snapshots) > 1:
                total_time = self._snapshots[-1].t - self._snapshots[0].t
                duration_ms = int((total_time / len(frames)) * 1000)
            else:
                duration_ms = int(1000 / self._frame_rate)
            duration_ms = max(duration_ms, 20)

            iio.imwrite(path, frames, extension=".gif", duration=duration_ms, loop=0)
        else:
            iio.imwrite(path, frames, fps=self._frame_rate)

        print(f"Saved to {path}")

    def _add_element_to_plotter(self, plotter, tracked: TrackedElement, opacity: float) -> dict:
        """Add an element to an off-screen plotter for export."""
        from morphis.elements.surface import Surface
        from morphis.visuals.drawing.vectors import (
            _create_arrow_mesh,
            create_frame_mesh,
        )

        element = tracked.element
        actors = {"edges": None, "faces": None, "origin": None}

        if isinstance(element, Surface):
            # Surface mesh
            from pyvista import PolyData

            vertices = element.vertices.data
            faces = element.faces
            mesh = PolyData(vertices, faces)
            actors["edges"] = plotter.add_mesh(
                mesh,
                color=tracked.color,
                opacity=opacity,
                smooth_shading=True,
            )

        elif isinstance(element, Frame):
            # Frame: multiple arrows from origin with optional faces
            origin = tracked.extra.get("origin", zeros(3))
            origin_3d = self._project_point(origin)
            vecs_3d = self._project_vectors(element.data)

            edges_mesh, faces_mesh, origin_mesh = create_frame_mesh(
                origin_3d,
                vecs_3d,
                projection_axes=None,  # Already projected
                filled=tracked.extra.get("filled", False),
            )

            if edges_mesh is not None:
                actors["edges"] = plotter.add_mesh(
                    edges_mesh, color=tracked.color, opacity=opacity, smooth_shading=True
                )

            if faces_mesh is not None:
                actors["faces"] = plotter.add_mesh(
                    faces_mesh, color=tracked.color, opacity=opacity * 0.2, smooth_shading=True
                )

            if origin_mesh is not None:
                actors["origin"] = plotter.add_mesh(
                    origin_mesh, color=tracked.color, opacity=opacity, smooth_shading=True
                )

        elif isinstance(element, Vector) and element.grade == 1:
            # Single vector or collection
            origin = tracked.extra.get("origin", zeros(3))
            origin_3d = self._project_point(origin)

            if element.lot and element.lot != ():
                # Collection of vectors
                vecs_3d = self._project_vectors(element.data)
                origins = tracked.extra.get("origins", zeros((len(element.data), 3)))
                origins_3d = self._project_vectors(origins)

                from pyvista import merge

                arrow_meshes = []
                for o, d in zip(origins_3d, vecs_3d, strict=False):
                    mesh = _create_arrow_mesh(o, d)
                    if mesh is not None:
                        arrow_meshes.append(mesh)

                if arrow_meshes:
                    combined = merge(arrow_meshes)
                    actors["edges"] = plotter.add_mesh(
                        combined, color=tracked.color, opacity=opacity, smooth_shading=True
                    )
            else:
                # Single vector
                vec_3d = self._project_point(element.data)
                mesh = _create_arrow_mesh(origin_3d, vec_3d)
                if mesh is not None:
                    actors["edges"] = plotter.add_mesh(mesh, color=tracked.color, opacity=opacity, smooth_shading=True)

        return actors

    def _update_element_in_plotter(self, plotter, tracked: TrackedElement, actors: dict, opacity: float) -> None:
        """Update an element's geometry and opacity in the plotter."""
        from morphis.elements.surface import Surface
        from morphis.visuals.drawing.vectors import create_frame_mesh

        element = tracked.element

        if isinstance(element, Surface):
            # Update mesh vertices
            if actors["edges"] is not None:
                actors["edges"].mapper.GetInput().points = element.vertices.data
                actors["edges"].GetProperty().SetOpacity(opacity)

        elif isinstance(element, Frame):
            # Update frame geometry
            origin = tracked.extra.get("origin", zeros(3))
            origin_3d = self._project_point(origin)
            vecs_3d = self._project_vectors(element.data)

            edges_mesh, faces_mesh, _ = create_frame_mesh(
                origin_3d,
                vecs_3d,
                projection_axes=None,
                filled=tracked.extra.get("filled", False),
            )

            if actors["edges"] is not None and edges_mesh is not None:
                actors["edges"].mapper.SetInputData(edges_mesh)
                actors["edges"].GetProperty().SetOpacity(opacity)

            if actors["faces"] is not None and faces_mesh is not None:
                actors["faces"].mapper.SetInputData(faces_mesh)
                actors["faces"].GetProperty().SetOpacity(opacity * 0.2)

            if actors["origin"] is not None:
                actors["origin"].GetProperty().SetOpacity(opacity)

    def _compute_auto_camera(self) -> tuple[tuple | None, tuple | None]:
        """
        Compute reasonable camera position based on tracked elements' extent.

        Strategy:
        - Compute bounding box of all tracked objects across all snapshots
        - Position camera to see entire scene with margin
        - Default viewing angle: isometric-ish (positive x, negative y, positive z)

        Returns:
            (camera_position, focal_point) or (None, None) if no data
        """
        from numpy import array, max as np_max, min as np_min
        from numpy.linalg import norm

        if not self._snapshots:
            return (4.0, -3.0, 3.5), (0.0, 0.0, 0.0)

        # Collect all vertex positions across all snapshots
        all_points = []

        for snapshot in self._snapshots:
            for element_id, state in snapshot.states.items():
                if element_id not in self._elements:
                    continue

                tracked = self._elements[element_id]
                element = tracked.element

                # Get the data from the element
                if hasattr(element, "vertices"):
                    data = state.get("vertices", element.vertices.data)
                elif hasattr(element, "data"):
                    data = state.get("data", element.data)
                else:
                    continue

                origin = tracked.extra.get("origin", zeros(3))
                origin_3d = self._project_point(origin)
                all_points.append(origin_3d)

                # Add extent from data
                if isinstance(element, Frame):
                    vecs_3d = self._project_vectors(data)
                    for vec in vecs_3d:
                        all_points.append(origin_3d + vec)
                elif hasattr(data, "ndim") and data.ndim >= 1:
                    if data.ndim == 1:
                        # Single point/vector
                        pt = self._project_point(data)
                        all_points.append(origin_3d + pt)
                    else:
                        # Multiple points
                        for row in data:
                            pt = self._project_point(row)
                            all_points.append(pt)

        if not all_points:
            return (4.0, -3.0, 3.5), (0.0, 0.0, 0.0)

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

        camera_position = tuple(center + direction * distance)
        camera_focal = tuple(center)

        return camera_position, camera_focal

    # =========================================================================
    # Display
    # =========================================================================

    def show(self, interactive: bool = True) -> None:
        """
        Display the scene.

        Args:
            interactive: If True, allows user interaction
        """
        self._ensure_backend()
        _bring_window_to_front()
        self._backend.show(interactive=interactive)

    def render(self) -> None:
        """Render current frame without showing window."""
        self._ensure_backend()
        self._backend.render()

    def close(self) -> None:
        """Close the scene and clean up."""
        if self._backend_initialized:
            self._backend.close()
            self._backend_initialized = False


def _bring_window_to_front():
    """Bring window to front (macOS)."""
    if sys.platform == "darwin":
        try:
            from AppKit import NSApp, NSApplication

            NSApplication.sharedApplication()
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            pass
