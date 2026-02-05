"""
Scene - Unified Visualization Interface

Scene provides a single interface for both static and animated visualization.
Live mode shows the animation in real-time. Export mode re-runs the animation
to capture frames without delays.
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


class Scene:
    """
    Unified visualization interface for static and animated scenes.

    Live animation example:
        scene = Scene(theme="obsidian")
        scene.add(F, color=RED, filled=True)
        for t in times:
            F.data[...] = transform(t)
            scene.capture(t)
        scene.show()  # Wait for window close

    Static example:
        scene = Scene(theme="obsidian")
        scene.add(v, color=RED)
        scene.show()
    """

    def __init__(
        self,
        projection: tuple[int, ...] | None = None,
        theme: str | Theme = "obsidian",
        size: tuple[int, int] = (600, 600),  # SMALL_SQUARE
        frame_rate: int = 30,
        backend: str = "pyvista",
        show_basis: bool = True,
    ):
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
        self._effects: list[SceneEffect] = []
        self._live_start_time: float | None = None
        self._first_capture = True

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

        if representation is None:
            representation = self._default_representation(element)

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
            from morphis.visuals.drawing.vectors import create_frame_mesh

            origin = kwargs.get("origin", zeros(3))
            origin_3d = self._project_point(origin)
            vecs_3d = self._project_vectors(element.data)

            edges_mesh, faces_mesh, origin_mesh = create_frame_mesh(
                origin_3d,
                vecs_3d,
                projection_axes=None,
                filled=kwargs.get("filled", False),
            )

            mesh_types = []

            if edges_mesh is not None:
                bid = self._backend.add_mesh(
                    edges_mesh.points,
                    edges_mesh.faces,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                )
                backend_ids.append(bid)
                mesh_types.append("edges")

            if faces_mesh is not None:
                bid = self._backend.add_mesh(
                    faces_mesh.points,
                    faces_mesh.faces,
                    color=color,
                    opacity=opacity * 0.2,
                    smooth_shading=True,
                )
                backend_ids.append(bid)
                mesh_types.append("faces")

            if origin_mesh is not None:
                bid = self._backend.add_mesh(
                    origin_mesh.points,
                    origin_mesh.faces,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                )
                backend_ids.append(bid)
                mesh_types.append("origin")

            kwargs["_mesh_types"] = mesh_types

        elif isinstance(element, Vector):
            if element.grade == 1:
                data = element.data
                if element.lot and element.lot != ():
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
        """Set projection axes for nD -> 3D."""
        if len(axes) != 3:
            raise ValueError("Projection requires exactly 3 axes")

        self._projection = axes

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
        """Set camera clipping range."""
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
        """Add a light to the scene."""
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

    def fade_in(self, element: Element, t: float, duration: float) -> None:
        """Schedule a fade-in effect for an element."""
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

    def fade_out(self, element: Element, t: float, duration: float) -> None:
        """Schedule a fade-out effect for an element."""
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
            return 1.0

        active = [e for e in relevant if e.is_active(t)]

        if not active:
            past = [e for e in relevant if t > e.t_end]
            if past:
                latest = max(past, key=lambda e: e.t_end)
                return latest.evaluate(latest.t_end)
            return 0.0

        current = max(active, key=lambda e: e.t_start)
        return current.evaluate(t)

    # =========================================================================
    # Animation
    # =========================================================================

    def capture(self, t: float) -> None:
        """
        Render current state at time t (live mode).

        Shows window on first call, then syncs to real-time.
        """
        self._ensure_backend()

        # Show window on first capture
        if self._first_capture:
            self._backend.show(interactive=False)
            _bring_window_to_front()
            self._live_start_time = time_module.time()
            self._first_capture = False

        # Sync visuals with current element state
        self._sync_visuals(t)

        # Wait for real-time sync
        if self._live_start_time is not None:
            target_time = self._live_start_time + t
            while time_module.time() < target_time:
                if self._backend.is_closed():
                    return
                self._backend.process_events()
                time_module.sleep(0.001)

    def _sync_visuals(self, t: float) -> None:
        """Synchronize backend visuals with current element state."""
        from morphis.elements.surface import Surface

        for element_id, tracked in self._elements.items():
            element = tracked.element

            # Compute opacity (faces get 0.2 multiplier)
            base_opacity = self._compute_opacity(element_id, t) * tracked.opacity
            mesh_types = tracked.extra.get("_mesh_types", [])
            for idx, bid in enumerate(tracked.backend_ids):
                mesh_type = mesh_types[idx] if idx < len(mesh_types) else None
                if mesh_type == "faces":
                    self._backend.set_opacity(bid, base_opacity * 0.2)
                else:
                    self._backend.set_opacity(bid, base_opacity)

            if isinstance(element, Surface):
                if tracked.backend_ids:
                    self._backend.update_mesh(
                        tracked.backend_ids[0],
                        element.vertices.data,
                    )

            elif isinstance(element, Frame):
                from morphis.visuals.drawing.vectors import create_frame_mesh

                origin = tracked.extra.get("origin", zeros(3))
                origin_3d = self._project_point(origin)
                vecs_3d = self._project_vectors(element.data)

                edges_mesh, faces_mesh, _ = create_frame_mesh(
                    origin_3d,
                    vecs_3d,
                    projection_axes=None,
                    filled=tracked.extra.get("filled", False),
                )

                mesh_types = tracked.extra.get("_mesh_types", [])
                for idx, bid in enumerate(tracked.backend_ids):
                    mesh_type = mesh_types[idx] if idx < len(mesh_types) else None
                    actor = self._backend.get_actor(bid)
                    if actor is None:
                        continue

                    if mesh_type == "edges" and edges_mesh is not None:
                        actor.mapper.SetInputData(edges_mesh)
                    elif mesh_type == "faces" and faces_mesh is not None:
                        actor.mapper.SetInputData(faces_mesh)

            elif isinstance(element, Vector) and element.grade == 1:
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

    def show(self) -> None:
        """Wait for user to close window."""
        self._ensure_backend()

        if self._backend.is_closed():
            return

        # If we haven't shown the window yet, show it now
        if self._first_capture:
            self._backend.show(interactive=False)
            _bring_window_to_front()
            self._first_capture = False

        self._backend.wait_for_close()

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
