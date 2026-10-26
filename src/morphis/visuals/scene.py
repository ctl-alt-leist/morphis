"""
Scene - Unified Visualization Interface

The Scene draws what it is handed. It holds references to the user's elements,
and every capture(t) re-reads their current state and redraws them. All the
mathematics of what the elements are doing (rotations, evolutions, simulations)
happens outside the Scene; the only mathematics inside is depiction: projecting
higher-dimensional elements to 3D through the scene's View, and turning each
element into something drawable.

    Vector, grade 1    arrows from an origin
    Vector, grade 2    an oriented disk: area |B| in the bivector's plane, with a circulation arrow
    Vector, grade 3    an oriented ball: volume |T|, with its handedness on the equator
    Frame              its exact vectors as arrows, plus their parallelogram or
                       parallelepiped when filled
    Surface, VisualModel   the mesh, with vertices re-read every capture

Depictions are stateless: each frame is computed from the element alone, so a
smoothly changing element gives a smoothly changing picture.

Under the projection Π, points and everything built from them (arrow tails and
tips, the corners of a frame, mesh vertices) are mapped point by point, which
keeps straight edges exact under perspective too. A bare blade has no points
of its own beyond where it is attached, so it is carried into the picture by
the outermorphism of dΠ there and drawn by its invariants.

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

from numpy import array, stack, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from morphis.elements.base import Element
from morphis.elements.frame import Frame
from morphis.elements.metric import Metric
from morphis.elements.vector import Vector
from morphis.visuals.backends import get_backend
from morphis.visuals.drawing.blades import oriented_ball, oriented_disk
from morphis.visuals.projection import basis_labels
from morphis.visuals.projective import DEFAULT_AXES, Orthographic, Projection, View, resolve
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
    projection: View | Projection
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
        Elements of dimension greater than 3 are drawn through a View: the
        kind of projection and the three directions, by label, that become
        x, y, z. Each element resolves the view against its own metric, so an
        axis that is not one of its directions raises IndexError. Elements of
        dimension 3 or less are drawn in their own coordinates.

        scene = Scene(projection=Orthographic(axes=(1, 2, 3)))   # the default
        scene.set_projection(Perspective(axes=(1, 2, 3), distance=3.0))
        scene.set_projection(Stereographic(axes=(1, 2, 3), pole=4))

        A Projection to 3D may be given instead of a view, for a map no view
        names, such as an orthographic screen tilted out of the basis.
    """

    def __init__(
        self,
        projection: View | Projection | None = None,
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
        self._projection = Orthographic() if projection is None else projection
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
    def projection(self) -> View | Projection:
        """The view, or given projection, that elements of dimension above 3 are drawn through."""
        return self._projection

    @property
    def basis_labels(self) -> tuple[str, ...]:
        """Labels for the drawn basis axes: the view's axes."""
        axes = self._projection.axes if isinstance(self._projection, View) else DEFAULT_AXES

        return basis_labels(axes)

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
            vertices = self._picture(element.vertices.metric)(element.vertices)
            shapes = {"mesh": Shape("mesh", {"vertices": vertices, "faces": element.faces})}

        elif isinstance(element, Frame):
            shapes = self._depict_frame(element, options)

        elif isinstance(element, Vector) and element.grade == 1:
            shapes = self._depict_vectors(element, options)

        elif isinstance(element, Vector) and element.grade in (2, 3):
            shapes = self._depict_oriented(element, options)

        else:
            raise ValueError(f"Scene cannot depict {type(element).__name__} of grade {getattr(element, 'grade', None)}")

        return shapes

    def _picture(self, metric: Metric) -> Projection:
        """The map from the element's space to the drawing space."""
        return resolve(self._projection, metric)

    def _origins(self, element: Element, options: dict, count: int) -> NDArray:
        """Where each lot entry (or the single element) is attached, in the element's own space."""
        is_lot = count > 1 or bool(element.lot)
        given = options.get("origins") if is_lot else options.get("origin")
        origins = zeros((count, element.dim)) if given is None else array(given, dtype=float).reshape(count, -1)

        return origins

    def _depict_vectors(self, element: Vector, options: dict) -> dict[str, Shape]:
        """Arrows from each origin, drawn as the segment between the images of the tail and the tip."""
        projection = self._picture(element.metric)
        vectors = element.data.reshape(-1, element.dim)
        origins = self._origins(element, options, len(vectors))
        tails = projection(origins)
        tips = projection(origins + vectors)

        return {"arrows": Shape("arrows", {"origins": tails, "directions": tips - tails})}

    def _depict_oriented(self, element: Vector, options: dict) -> dict[str, Shape]:
        """Oriented disks for bivectors, oriented balls for trivectors, one per lot entry."""
        grade = element.grade
        projection = self._picture(element.metric)
        entries = element.data.reshape(-1, *element.data.shape[-grade:])
        origins = self._origins(element, options, len(entries))

        shapes = {}
        for n, (components, origin) in enumerate(zip(entries, origins, strict=True)):
            blade = Vector(components, grade=grade, metric=element.metric)
            drawn = projection.blade(blade, at=origin)
            center = projection(origin)
            depiction = oriented_disk(drawn, center) if grade == 2 else oriented_ball(drawn[0, 1, 2], center)
            shapes[f"fill/{n}"] = Shape("mesh", {"vertices": depiction.surface, "faces": depiction.faces}, FILL_OPACITY)
            shapes[f"rim/{n}"] = Shape("line", {"points": depiction.rim})
            shapes[f"sense/{n}"] = Shape("mesh", {"vertices": depiction.head, "faces": depiction.head_faces})

        return shapes

    def _depict_frame(self, element: Frame, options: dict) -> dict[str, Shape]:
        """The frame's exact vectors, and the shape they span when filled, from the images of its corners."""
        from morphis.visuals.drawing.vectors import create_frame_mesh

        origin = self._origins(element, options, 1)[0]
        vectors = element.data.reshape(-1, element.dim)

        # Corner n is the origin plus vector j for every bit j set in n
        k = len(vectors)
        corners = stack([origin + array([(n >> j) & 1 for j in range(k)]) @ vectors for n in range(2**k)])
        drawn = self._picture(element.metric)(corners)
        edges, faces, marker = create_frame_mesh(drawn, filled=options.get("filled", False))

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

    # =========================================================================
    # Projection
    # =========================================================================

    def set_projection(self, projection: View | Projection) -> None:
        """
        Draw elements of dimension above 3 through a new view, or a given projection.

        Args:
            projection: An Orthographic, Perspective, or Stereographic view, or
                a Projection to 3D. The basis labels follow the view's axes.

        Raises:
            IndexError: if an axis or pole is not a direction of an element
                already in the scene.
            ValueError: if a given Projection cannot draw such an element.

        Every element already in the scene is redrawn through it.
        """
        # Validate against every projected element before changing state
        for tracked in self._elements.values():
            metric = getattr(tracked.element, "metric", None)
            if metric is not None and metric.dim > 3:
                resolve(projection, metric)

        self._projection = projection

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
