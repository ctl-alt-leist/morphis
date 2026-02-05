"""
Render Backend Protocol

Abstract interface for visualization backends. The Scene class uses this
protocol to render elements, allowing different backends (PyVista, matplotlib,
plotly, etc.) to be swapped without changing the Scene API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from numpy.typing import NDArray


if TYPE_CHECKING:
    from morphis.visuals.theme import Color, Theme


@runtime_checkable
class RenderBackend(Protocol):
    """
    Protocol for visualization backends.

    Backends manage a rendering window and provide methods to add, update,
    and remove visual objects. All geometric data is passed as numpy arrays.

    Object IDs are strings returned by add_* methods and used to reference
    objects for updates or removal.
    """

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(
        self,
        size: tuple[int, int],
        theme: "Theme",
        show_basis: bool = True,
    ) -> None:
        """
        Initialize the rendering window.

        Args:
            size: Window size (width, height)
            theme: Visual theme for colors
            show_basis: Whether to show coordinate basis
        """
        ...

    def close(self) -> None:
        """Close the rendering window and clean up resources."""
        ...

    # =========================================================================
    # Mesh Operations
    # =========================================================================

    def add_mesh(
        self,
        vertices: NDArray,
        faces: NDArray,
        color: "Color",
        opacity: float = 1.0,
        smooth_shading: bool = True,
        show_edges: bool = False,
    ) -> str:
        """
        Add a mesh to the scene.

        Args:
            vertices: Vertex positions (N, 3)
            faces: Face connectivity in indexed format [n, v0, v1, ..., n, ...]
            color: RGB color tuple
            opacity: Opacity [0, 1]
            smooth_shading: Use smooth shading
            show_edges: Show wireframe edges

        Returns:
            Object ID for later reference
        """
        ...

    def update_mesh(
        self,
        object_id: str,
        vertices: NDArray,
    ) -> None:
        """
        Update mesh vertex positions.

        Args:
            object_id: ID from add_mesh
            vertices: New vertex positions (N, 3)
        """
        ...

    # =========================================================================
    # Arrow Operations
    # =========================================================================

    def add_arrows(
        self,
        origins: NDArray,
        directions: NDArray,
        color: "Color",
        opacity: float = 1.0,
        tip_length: float = 0.1,
        tip_radius: float = 0.03,
        shaft_radius: float = 0.015,
    ) -> str:
        """
        Add arrows to the scene.

        Args:
            origins: Arrow start points (N, 3) or (3,) for single arrow
            directions: Arrow vectors (N, 3) or (3,) for single arrow
            color: RGB color tuple
            opacity: Opacity [0, 1]
            tip_length: Arrowhead length ratio
            tip_radius: Arrowhead radius
            shaft_radius: Arrow shaft radius

        Returns:
            Object ID for later reference
        """
        ...

    def update_arrows(
        self,
        object_id: str,
        origins: NDArray,
        directions: NDArray,
    ) -> None:
        """
        Update arrow positions and directions.

        Args:
            object_id: ID from add_arrows
            origins: New arrow start points
            directions: New arrow vectors
        """
        ...

    # =========================================================================
    # Point Operations
    # =========================================================================

    def add_points(
        self,
        positions: NDArray,
        color: "Color",
        opacity: float = 1.0,
        point_size: float = 5.0,
    ) -> str:
        """
        Add points to the scene.

        Args:
            positions: Point positions (N, 3)
            color: RGB color tuple
            opacity: Opacity [0, 1]
            point_size: Point size in pixels

        Returns:
            Object ID for later reference
        """
        ...

    def update_points(
        self,
        object_id: str,
        positions: NDArray,
    ) -> None:
        """
        Update point positions.

        Args:
            object_id: ID from add_points
            positions: New point positions
        """
        ...

    # =========================================================================
    # Line Operations
    # =========================================================================

    def add_lines(
        self,
        points: NDArray,
        color: "Color",
        opacity: float = 1.0,
        line_width: float = 2.0,
    ) -> str:
        """
        Add a polyline to the scene.

        Args:
            points: Line vertices (N, 3) connected in order
            color: RGB color tuple
            opacity: Opacity [0, 1]
            line_width: Line width in pixels

        Returns:
            Object ID for later reference
        """
        ...

    def update_lines(
        self,
        object_id: str,
        points: NDArray,
    ) -> None:
        """
        Update line vertices.

        Args:
            object_id: ID from add_lines
            points: New line vertices
        """
        ...

    # =========================================================================
    # Span (Bivector) Operations
    # =========================================================================

    def add_span(
        self,
        origin: NDArray,
        vectors: NDArray,
        color: "Color",
        opacity: float = 0.3,
        filled: bool = True,
    ) -> str:
        """
        Add a span (parallelogram/parallelepiped) to the scene.

        For grade-2: parallelogram from 2 vectors
        For grade-3: parallelepiped from 3 vectors

        Args:
            origin: Origin point (3,)
            vectors: Spanning vectors (k, 3) where k is 2 or 3
            color: RGB color tuple
            opacity: Opacity [0, 1]
            filled: Show filled faces (vs wireframe only)

        Returns:
            Object ID for later reference
        """
        ...

    def update_span(
        self,
        object_id: str,
        origin: NDArray,
        vectors: NDArray,
    ) -> None:
        """
        Update span geometry.

        Args:
            object_id: ID from add_span
            origin: New origin
            vectors: New spanning vectors
        """
        ...

    # =========================================================================
    # Text Operations
    # =========================================================================

    def add_text(
        self,
        text: str,
        position: NDArray,
        color: "Color",
        font_size: int = 12,
        anchor: str = "center",
    ) -> str:
        """
        Add 3D text annotation to the scene.

        Args:
            text: Text content
            position: 3D position (3,)
            color: RGB color tuple
            font_size: Font size
            anchor: Text anchor ("center", "left", "right", etc.)

        Returns:
            Object ID for later reference
        """
        ...

    def update_text(
        self,
        object_id: str,
        text: str | None = None,
        position: NDArray | None = None,
    ) -> None:
        """
        Update text content and/or position.

        Args:
            object_id: ID from add_text
            text: New text content (None = keep current)
            position: New position (None = keep current)
        """
        ...

    # =========================================================================
    # Object Management
    # =========================================================================

    def set_opacity(self, object_id: str, opacity: float) -> None:
        """
        Set object opacity.

        Args:
            object_id: Object ID
            opacity: New opacity [0, 1]
        """
        ...

    def remove(self, object_id: str) -> None:
        """
        Remove an object from the scene.

        Args:
            object_id: Object ID to remove
        """
        ...

    # =========================================================================
    # Camera
    # =========================================================================

    def set_camera(
        self,
        position: tuple[float, float, float] | None = None,
        focal_point: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Set camera position and orientation.

        Args:
            position: Camera position (None = keep current)
            focal_point: Look-at point (None = keep current)
            up: Up vector (None = keep current)
        """
        ...

    def reset_camera(self) -> None:
        """Reset camera to fit all objects in view."""
        ...

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(self) -> None:
        """Render the current frame."""
        ...

    def capture_frame(self) -> NDArray:
        """
        Capture current frame as image.

        Returns:
            RGB image array (H, W, 3)
        """
        ...

    def show(self, interactive: bool = True) -> None:
        """
        Display the scene.

        Args:
            interactive: Allow user interaction (vs immediate return)
        """
        ...

    def process_events(self) -> None:
        """Process pending window events to keep UI responsive during animations."""
        ...

    def is_closed(self) -> bool:
        """Check if the window has been closed by the user."""
        ...

    def wait_for_close(self) -> None:
        """Block until user closes window, while keeping it responsive."""
        ...

    # =========================================================================
    # Basis Display
    # =========================================================================

    def set_basis_labels(self, labels: tuple[str, str, str]) -> None:
        """
        Update coordinate basis labels.

        Args:
            labels: Labels for (x, y, z) axes
        """
        ...

    # =========================================================================
    # Lighting
    # =========================================================================

    def add_light(
        self,
        position: tuple[float, float, float] = (1, 1, 1),
        focal_point: tuple[float, float, float] = (0, 0, 0),
        intensity: float = 1.0,
        color: "Color" = (1.0, 1.0, 1.0),
        directional: bool = True,
        attenuation: tuple[float, float, float] | None = None,
    ) -> str:
        """
        Add a light to the scene.

        Args:
            position: Light position (or direction for directional lights)
            focal_point: Point the light aims at
            intensity: Light brightness [0, inf)
            color: Light color RGB
            directional: If True, parallel rays (like sun at infinity).
                        If False, point light with position.
            attenuation: For positional lights: (constant, linear, quadratic).
                        Intensity falls as 1/(c + l*d + q*d²).
                        None = no falloff. (1, 0, 0) = constant.
                        (0, 0, 1) = inverse square law.

        Returns:
            Light ID for later reference
        """
        ...

    def remove_light(self, light_id: str) -> None:
        """Remove a light from the scene."""
        ...

    def clear_lights(self) -> None:
        """Remove all lights from the scene."""
        ...
