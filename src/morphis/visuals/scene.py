"""
Scene - Unified Visualization Interface

The Scene draws what it is handed. It holds references to the user's elements,
and every capture(t) re-reads their current state and redraws them. All the
mathematics of what the elements are doing (rotations, evolutions, simulations)
happens outside the Scene; the only mathematics inside is depiction: projecting
higher-dimensional elements to 3D, and turning each element into something
drawable.

    Vector, grade 1    arrows from an origin
    Vector, grade 2    an oriented disk: area |B| in the bivector's plane, with a circulation arrow
    Vector, grade 3    an oriented ball: volume |T|, with its handedness on the equator
    Frame              its exact vectors as arrows, plus their parallelogram or
                       parallelepiped when filled
    Surface, VisualModel   the mesh, with vertices re-read every capture

Depictions are stateless: each frame is computed from the element alone, so a
smoothly changing element gives a smoothly changing picture.

Recording wraps the capture loop:

    with scene.record("figures/orbit/orbit.mp4"):
        for t in times:
            ...  # update elements
            scene.capture(t)

With window=False the Scene renders off screen and does not wait for wall-clock
time, so recordings run as fast as frames render.
"""

from __future__ import annotations

import pickle
import sys
import time as time_module
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from numpy import array, ix_, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.elements.base import Element
from morphis.elements.frame import Frame
from morphis.elements.metric import Metric
from morphis.elements.vector import Vector
from morphis.visuals.backends import get_backend
from morphis.visuals.drawing.blades import oriented_ball, oriented_disk, pad_to_3d
from morphis.visuals.projection import (
    DEFAULT_PROJECTION,
    basis_labels,
    projection_slots,
    validate_projection_axes,
)
from morphis.visuals.recording import Recording
from morphis.visuals.theme import Color, Theme, get_theme


if TYPE_CHECKING:
    from morphis.visuals.backends.protocol import RenderBackend


# Opacity of filled surfaces relative to their element
FILL_OPACITY = 0.3
FRAME_FACE_OPACITY = 0.2


@dataclass
class SceneData:
    """Serializable scene state for save/load."""

    theme_name: str
    size: tuple[int, int]
    projection: tuple[int, ...]  # User-facing geometric indices
    show_basis: bool
    elements: list[dict]


@dataclass
class Shape:
    """
    One drawable piece of an element's depiction.

    kind selects the backend primitive: "arrows" (origins, directions), "mesh"
    (vertices with fixed faces), "reshaped" (vertices and faces that may both
    change), or "line" (a polyline).
    """

    kind: str
    geometry: dict[str, NDArray]
    opacity: float = 1.0


@dataclass
class Part:
    """A backend object drawing one Shape."""

    backend_id: str
    kind: str
    opacity: float


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
    parts: dict[str, Any]  # shape name -> Part
    color: Color
    opacity: float
    extra: dict  # Placement and style options given to add()


class Scene:
    """
    Unified visualization interface for static, live, and recorded scenes.

    The Scene draws what it is handed: mutate your elements, then capture.

    Live animation:
        scene = Scene(theme="obsidian")
        scene.add(F, color=RED, filled=True)
        for t in times:
            F.data[...] = transform(t)
            scene.capture(t)
        scene.show()  # Wait for window close

    Recording (off screen, as fast as frames render):
        scene = Scene(window=False)
        scene.add(B)
        with scene.record("figures/turn/turn.gif"):
            for t in times:
                B.data[...] = turned(t)
                scene.capture(t)

    Static:
        scene = Scene(theme="obsidian")
        scene.add(v, color=RED)
        scene.show()

    Projection:
        Elements of dimension greater than 3 are projected onto three axes
        given as user-facing geometric indices, the same indices used by
        basis_vector and .on[...]. The default (1, 2, 3) is x, y, z in every
        signature. Each element translates the axes through its own metric,
        so an out-of-range index (0 in a Euclidean metric) raises IndexError.
        Elements of dimension 3 or less are drawn in their own coordinates.

        scene = Scene(projection=(1, 2, 3))  # e_1, e_2, e_3
        scene.set_projection((2, 3, 4))      # e_2, e_3, e_4
    """

    def __init__(
        self,
        projection: tuple[int, ...] | None = None,
        theme: str | Theme = "obsidian",
        size: tuple[int, int] = (1280, 800),  # MEDIUM
        frame_rate: int = 30,
        backend: str = "pyvista",
        show_basis: bool = True,
        auto_camera: bool = True,
        window: bool = True,
    ):
        if isinstance(theme, str):
            theme = get_theme(theme)

        self._theme = theme
        self._size = size
        self._frame_rate = frame_rate
        self._projection = DEFAULT_PROJECTION if projection is None else validate_projection_axes(projection)
        self._show_basis = show_basis
        self._auto_camera = auto_camera
        self._window = window

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
        self._recording: Recording | None = None

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

    @property
    def window(self) -> bool:
        """Whether the scene renders to an on-screen window (False renders off screen)."""
        return self._window

    @property
    def projection(self) -> tuple[int, int, int]:
        """Projection axes as user-facing geometric indices."""
        return self._projection

    @property
    def basis_labels(self) -> tuple[str, ...]:
        """Labels for the drawn basis axes, derived from the projection axes."""
        return basis_labels(self._projection)

    # =========================================================================
    # Backend Management
    # =========================================================================

    def _ensure_backend(self):
        """Initialize backend if needed."""
        if not self._backend_initialized:
            # Labels set before initialize are used when the basis is first drawn
            self._backend.set_basis_labels(self.basis_labels)
            self._backend.initialize(
                size=self._size,
                theme=self._theme,
                show_basis=self._show_basis,
                window=self._window,
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
        color: Color | None = None,
        opacity: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Add an element to the scene.

        Args:
            element: Vector (grade 1, 2, or 3), Frame, Surface, or VisualModel
            color: RGB color tuple (uses palette if None)
            opacity: Opacity [0, 1]
            **kwargs: Placement and style options:
                origin: Where a single vector, disk, ball, or frame is drawn from
                origins: One origin per lot entry, for elements with a lot
                filled: For a Frame, also draw the parallelogram or parallelepiped
                smooth_shading, show_edges: For meshes

        Returns:
            Element ID for later reference
        """
        self._ensure_backend()

        element_id = str(uuid4())
        color = color if color is not None else self._next_color()

        parts = {}
        for name, shape in self._depict(element, kwargs).items():
            backend_id = self._create_part(shape, color, opacity, kwargs)
            parts[name] = Part(backend_id=backend_id, kind=shape.kind, opacity=shape.opacity)

        self._elements[element_id] = TrackedElement(
            element_id=element_id,
            element=element,
            parts=parts,
            color=color,
            opacity=opacity,
            extra=kwargs,
        )

        return element_id

    def remove(self, element_id: str) -> None:
        """Remove an element from the scene."""
        if element_id not in self._elements:
            return

        tracked = self._elements.pop(element_id)
        for part in tracked.parts.values():
            self._backend.remove(part.backend_id)

    # -------------------------------------------------------------------------
    # Depiction: element state -> shapes (stateless)
    # -------------------------------------------------------------------------

    def _depict(self, element: Element, options: dict) -> dict[str, Shape]:
        """
        Depict an element's current state as named shapes.

        The names and kinds depend only on the element's type, grade, and lot,
        so the same element always yields the same set of shapes, frame after
        frame; only the geometry changes.
        """
        from morphis.elements.surface import Surface
        from morphis.visuals.model import VisualModel

        if isinstance(element, (Surface, VisualModel)):
            shapes = {"mesh": Shape("mesh", {"vertices": element.vertices.data, "faces": element.faces})}

        elif isinstance(element, Frame):
            shapes = self._depict_frame(element, options)

        elif isinstance(element, Vector) and element.grade == 1:
            shapes = self._depict_vectors(element, options)

        elif isinstance(element, Vector) and element.grade in (2, 3):
            shapes = self._depict_oriented(element, options)

        else:
            raise ValueError(f"Scene cannot depict {type(element).__name__} of grade {getattr(element, 'grade', None)}")

        return shapes

    def _origins(self, element: Element, options: dict, count: int) -> NDArray:
        """Projected origins for each lot entry (or the single element)."""
        is_lot = count > 1 or bool(element.lot)
        given = options.get("origins") if is_lot else options.get("origin")
        raw = zeros((count, 3)) if given is None else array(given, dtype=float).reshape(count, -1)
        origins = array([self._project_point(point, element.metric) for point in raw])

        return origins

    def _depict_vectors(self, element: Vector, options: dict) -> dict[str, Shape]:
        directions = self._project_vectors(element.data, element.metric)
        origins = self._origins(element, options, len(directions))

        return {"arrows": Shape("arrows", {"origins": origins, "directions": directions})}

    def _depict_oriented(self, element: Vector, options: dict) -> dict[str, Shape]:
        """Oriented disks for bivectors, oriented balls for trivectors, one per lot entry."""
        grade = element.grade
        entries = element.data.reshape(-1, *element.data.shape[-grade:])
        centers = self._origins(element, options, len(entries))

        shapes = {}
        for n, (components, center) in enumerate(zip(entries, centers, strict=True)):
            projected = self._project_components(components, grade, element.metric)
            depiction = oriented_disk(projected, center) if grade == 2 else oriented_ball(projected[0, 1, 2], center)
            shapes[f"fill/{n}"] = Shape("mesh", {"vertices": depiction.surface, "faces": depiction.faces}, FILL_OPACITY)
            shapes[f"rim/{n}"] = Shape("line", {"points": depiction.rim})
            shapes[f"sense/{n}"] = Shape("mesh", {"vertices": depiction.head, "faces": depiction.head_faces})

        return shapes

    def _depict_frame(self, element: Frame, options: dict) -> dict[str, Shape]:
        """The frame's exact vectors, and the shape they span when filled."""
        from morphis.visuals.drawing.vectors import create_frame_mesh

        origin = self._origins(element, options, 1)[0]
        vectors = self._project_vectors(element.data, element.metric)
        edges, faces, marker = create_frame_mesh(origin, vectors, filled=options.get("filled", False))

        meshes = {"edges": (edges, 1.0), "faces": (faces, FRAME_FACE_OPACITY), "origin": (marker, 1.0)}
        shapes = {
            name: Shape("reshaped", {"vertices": mesh.points, "faces": mesh.faces}, opacity)
            for name, (mesh, opacity) in meshes.items()
            if mesh is not None
        }

        return shapes

    # -------------------------------------------------------------------------
    # Backend parts
    # -------------------------------------------------------------------------

    def _create_part(self, shape: Shape, color: Color, opacity: float, options: dict) -> str:
        """Create the backend object for a shape."""
        geometry = shape.geometry
        alpha = opacity * shape.opacity

        if shape.kind == "arrows":
            backend_id = self._backend.add_arrows(
                geometry["origins"], geometry["directions"], color=color, opacity=alpha
            )

        elif shape.kind == "line":
            backend_id = self._backend.add_lines(geometry["points"], color=color, opacity=alpha)

        else:
            backend_id = self._backend.add_mesh(
                geometry["vertices"],
                geometry["faces"],
                color=color,
                opacity=alpha,
                smooth_shading=options.get("smooth_shading", True),
                show_edges=options.get("show_edges", False),
            )

        return backend_id

    def _update_part(self, part: Part, shape: Shape) -> None:
        """Move a backend object to a shape's current geometry."""
        geometry = shape.geometry

        if part.kind == "arrows":
            self._backend.update_arrows(part.backend_id, geometry["origins"], geometry["directions"])

        elif part.kind == "line":
            self._backend.update_lines(part.backend_id, geometry["points"])

        elif part.kind == "reshaped":
            self._backend.replace_mesh(part.backend_id, geometry["vertices"], geometry["faces"])

        else:
            self._backend.update_mesh(part.backend_id, geometry["vertices"])

    def _redraw(self, tracked: TrackedElement) -> None:
        """Redraw an element from its current state."""
        shapes = self._depict(tracked.element, tracked.extra)
        for name, part in tracked.parts.items():
            if name in shapes:
                self._update_part(part, shapes[name])

    # -------------------------------------------------------------------------
    # Projection of components
    # -------------------------------------------------------------------------

    def _project_point(self, point: NDArray, metric: Metric) -> NDArray:
        """
        Project a point to 3D using the current projection axes.

        Points of dimension 3 or less are zero-padded and drawn in their own
        coordinates. Higher-dimensional points take the components at the
        projection axes, translated to storage slots through the metric.
        """
        point = array(point, dtype=float)
        if len(point) <= 3:
            result = zeros(3)
            result[: len(point)] = point
        else:
            slots = projection_slots(self._projection, metric)
            result = point[list(slots)]

        return result

    def _project_vectors(self, vectors: NDArray, metric: Metric) -> NDArray:
        """Project vectors of shape (..., dim) to an array of shape (k, 3)."""
        vectors = array(vectors, dtype=float)
        flat = vectors.reshape(-1, vectors.shape[-1])
        result = array([self._project_point(v, metric) for v in flat])

        return result

    def _project_components(self, components: NDArray, grade: int, metric: Metric) -> NDArray:
        """Project the components of one grade-2 or grade-3 element to 3D."""
        if metric.dim <= 3:
            result = pad_to_3d(components, grade)
        else:
            slots = list(projection_slots(self._projection, metric))
            result = components[ix_(*([slots] * grade))]

        return result

    # =========================================================================
    # Projection
    # =========================================================================

    def set_projection(self, axes: tuple[int, ...]) -> None:
        """
        Set projection axes for nD -> 3D.

        Args:
            axes: Three user-facing geometric indices, e.g. (2, 3, 4) to show
                e_2, e_3, e_4. The basis labels follow the same indices.

        Raises:
            ValueError: if axes is not exactly three integers.
            IndexError: if an axis is out of range for the metric of any
                element of dimension greater than 3 already in the scene.

        Every element already in the scene is redrawn under the new axes.
        """
        axes = validate_projection_axes(axes)

        # Validate against every projected element before changing state
        for tracked in self._elements.values():
            metric = getattr(tracked.element, "metric", None)
            if metric is not None and metric.dim > 3:
                projection_slots(axes, metric)

        self._projection = axes

        if self._backend_initialized:
            self._backend.set_basis_labels(self.basis_labels)
            for tracked in self._elements.values():
                self._redraw(tracked)
            self._backend.render()

    # =========================================================================
    # Camera
    # =========================================================================

    def camera(
        self,
        position: tuple[float, float, float] | None = None,
        focal_point: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
    ) -> None:
        """Set camera position and orientation. Disables auto_camera."""
        self._auto_camera = False
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
    # Animation and Recording
    # =========================================================================

    def capture(self, t: float) -> None:
        """
        Redraw every element from its current state at time t.

        With a window, the first capture opens it and later captures keep pace
        with wall-clock time. Off screen, captures run as fast as they render.
        Inside record(), each capture also appends one frame.
        """
        self._ensure_backend()

        if self._window and self._backend.is_closed():
            return

        if self._first_capture:
            self._open()
            self._live_start_time = time_module.time()

        self._sync_visuals(t)

        if self._recording is not None:
            self._recording.append(self._backend.capture_frame())

        # Keep pace with wall-clock time when shown live
        if self._window and self._live_start_time is not None:
            target_time = self._live_start_time + t
            while time_module.time() < target_time:
                if self._backend.is_closed():
                    return
                self._backend.process_events()
                time_module.sleep(0.001)

    @contextmanager
    def record(self, path: str | Path, frame_rate: float | None = None) -> Iterator[Recording]:
        """
        Record every capture inside the block to a video or GIF.

        The suffix picks the format: .mp4 (H.264) or .gif (looping). The file
        is finalized when the block exits.

            with scene.record("figures/orbit/orbit.mp4"):
                for t in times:
                    ...
                    scene.capture(t)

        Args:
            path: Output file, ending in .mp4 or .gif
            frame_rate: Frames per second; defaults to the scene's frame rate

        Yields:
            The Recording, whose frame_count tells how many frames were written
        """
        recording = Recording(path, frame_rate or self._frame_rate)
        self._recording = recording
        try:
            yield recording
        finally:
            self._recording = None
            recording.close()

    def _open(self) -> None:
        """First render: open the window (if any) and frame the camera."""
        if self._window:
            self._backend.show(interactive=False)
            _bring_window_to_front()
        if self._auto_camera:
            self._backend.reset_camera()
        self._first_capture = False

    def _sync_visuals(self, t: float) -> None:
        """Redraw every element from its current state, with its opacity at time t."""
        for element_id, tracked in self._elements.items():
            self._redraw(tracked)

            base_opacity = self._compute_opacity(element_id, t) * tracked.opacity
            for part in tracked.parts.values():
                self._backend.set_opacity(part.backend_id, base_opacity * part.opacity)

        self._backend.render()

    def show(self) -> None:
        """Show the window and wait for the user to close it; an off-screen scene has nothing to show."""
        self._ensure_backend()

        if self._window and not self._backend.is_closed():
            if self._first_capture:
                self._open()
            self._backend.wait_for_close()

    def close(self) -> None:
        """Close the scene and clean up."""
        if self._backend_initialized:
            self._backend.close()
            self._backend_initialized = False

    def is_closed(self) -> bool:
        """Check if the window has been closed."""
        if not self._backend_initialized:
            return True
        return self._backend.is_closed()

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Save scene to file.

        Supported formats (determined by extension):
            .scene - Pickle format, reloadable with Scene.load()
            .obj   - Wavefront OBJ, viewable in macOS Preview and 3D apps

        Args:
            path: File path with extension (.scene or .obj)

        Example:
            scene.save("my_scene.scene")  # Reloadable
            scene.save("my_scene.obj")    # For Preview/3D apps
        """
        path = Path(path).expanduser()
        ext = path.suffix.lower()

        if ext == ".scene":
            self._save_scene(path)
        elif ext == ".obj":
            self._save_obj(path)
        else:
            raise ValueError(f"Unknown format '{ext}'. Use .scene or .obj")

    def _save_scene(self, path: Path) -> None:
        """Save as pickle (.scene format)."""
        data = SceneData(
            theme_name=self._theme.name,
            size=self._size,
            projection=self._projection,
            show_basis=self._show_basis,
            elements=[
                {
                    "element": t.element,
                    "color": t.color,
                    "opacity": t.opacity,
                    "extra": t.extra,
                }
                for t in self._elements.values()
            ],
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _save_obj(self, path: Path) -> None:
        """Save as Wavefront OBJ."""
        self._ensure_backend()
        self._backend.export_obj(str(path))

    @classmethod
    def load(cls, path: str | Path, window: bool = True) -> Scene:
        """
        Load a scene from a .scene file.

        Args:
            path: Path to the .scene file
            window: Open on screen (the default) or render off screen

        Returns:
            Scene ready to display with show()

        Example:
            scene = Scene.load("my_scene.scene")
            scene.show()
        """
        path = Path(path).expanduser()
        with open(path, "rb") as f:
            data: SceneData = pickle.load(f)

        scene = cls(
            theme=data.theme_name,
            size=data.size,
            projection=data.projection,
            show_basis=data.show_basis,
            window=window,
        )

        for elem_data in data.elements:
            scene.add(
                elem_data["element"],
                color=elem_data["color"],
                opacity=elem_data["opacity"],
                **elem_data["extra"],
            )

        return scene


def _bring_window_to_front():
    """Bring window to front (macOS)."""
    if sys.platform == "darwin":
        try:
            from AppKit import NSApp, NSApplication

            NSApplication.sharedApplication()
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            pass
