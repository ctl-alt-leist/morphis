"""
3D Visualization Canvas

PyVista-based canvas for rendering geometric objects with full control over
appearance. Provides clean arrow primitives with proper proportions, smooth
curves, and toggleable basis vectors.

The canvas tracks a color index into the theme's palette, automatically
cycling through colors as objects are added. Colors can also be specified
explicitly for full control.
"""

from numpy import array, cross, ndarray
from numpy.linalg import norm as np_norm
from pydantic import BaseModel, ConfigDict

from morphis.visuals.theme import DEFAULT_THEME, Color, Palette, Theme, get_theme


# =============================================================================
# Canvas Configuration
# =============================================================================


class ArrowStyle(BaseModel):
    """Configuration for arrow rendering proportions."""

    model_config = ConfigDict(frozen=True)

    shaft_ratio: float = 0.85
    shaft_radius: float = 0.012
    tip_radius_factor: float = 2.5
    resolution: int = 24


class CurveStyle(BaseModel):
    """Configuration for curve rendering."""

    model_config = ConfigDict(frozen=True)

    radius: float = 0.008
    resolution: int = 24
    spline_factor: int = 3


# =============================================================================
# Canvas
# =============================================================================


class Canvas:
    """
    3D visualization canvas with theme support and automatic color cycling.

    The canvas wraps PyVista's Plotter with a cleaner API focused on geometric
    objects. Colors cycle through the theme's palette unless explicitly specified.

    Example:
        canvas = Canvas("obsidian")
        canvas.arrow([0, 0, 0], [1, 0, 0])  # Uses first palette color
        canvas.arrow([0, 0, 0], [0, 1, 0])  # Uses second palette color
        canvas.arrow([0, 0, 0], [0, 0, 1], color=(1, 0, 0))  # Explicit red
        canvas.show()
    """

    def __init__(
        self,
        theme: str | Theme = DEFAULT_THEME,
        title: str | None = None,
        size: tuple[int, int] = (1200, 900),
        show_basis: bool = True,
        basis_axes: tuple[int, int, int] = (0, 1, 2),
    ):
        import pyvista as pv

        if isinstance(theme, str):
            theme = get_theme(theme)

        self.theme = theme
        self._title = title
        self._size = size
        self._color_index = 0
        self._basis_axes = basis_axes

        self.arrow_style = ArrowStyle()
        self.curve_style = CurveStyle()

        self.plotter = pv.Plotter(off_screen=False)
        self.plotter.set_background(theme.background)
        self.plotter.window_size = size

        if show_basis:
            self.basis(axes=basis_axes)

    def _next_color(self) -> Color:
        """Get next color from palette and advance index."""
        color = self.theme.palette[self._color_index]
        self._color_index += 1

        return color

    def _resolve_color(self, color: Color | None) -> Color:
        """Resolve color: use provided or get next from palette."""
        if color is None:
            return self._next_color()

        return color

    # -------------------------------------------------------------------------
    # Basis Vectors
    # -------------------------------------------------------------------------

    def basis(
        self,
        scale: float = 1.0,
        axes: tuple[int, int, int] = (0, 1, 2),
        labels: bool = True,
    ):
        """
        Draw coordinate axes at origin using PyVista's native axes.

        Args:
            scale: Not used for native axes (kept for API compatibility)
            axes: Which basis vectors to label (0-indexed, e.g., (0, 1, 2) or (1, 3, 4))
            labels: Whether to show axis labels
        """
        color = self.theme.axis_color

        # Generate labels based on which axes are displayed (0-indexed to 1-indexed for display)
        xlabel = f"e{axes[0] + 1}" if len(axes) > 0 else "e1"
        ylabel = f"e{axes[1] + 1}" if len(axes) > 1 else "e2"
        zlabel = f"e{axes[2] + 1}" if len(axes) > 2 else "e3"

        self.plotter.add_axes_at_origin(
            x_color=color,
            y_color=color,
            z_color=color,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            line_width=1,
            labels_off=not labels,
        )

        # Store current axes for reference
        self._basis_axes = axes

    def set_basis_axes(self, axes: tuple[int, int, int]):
        """
        Change which basis vectors are displayed.

        Useful for higher-dimensional spaces where we project onto
        different 3D subspaces. Updates axis labels to reflect the
        selected dimensions.

        Args:
            axes: Tuple of axis indices (0-indexed), e.g., (0, 2, 4) for e1, e3, e5
        """
        self.plotter.clear()
        self.plotter.set_background(self.theme.background)
        self.basis(axes=axes)

    # -------------------------------------------------------------------------
    # Arrow Primitive
    # -------------------------------------------------------------------------

    def _add_arrow_geometry(
        self,
        start: ndarray,
        direction: ndarray,
        color: Color,
        shaft_radius: float | None = None,
    ):
        """Internal: add arrow mesh to plotter."""
        import pyvista as pv

        start = array(start, dtype=float)
        direction = array(direction, dtype=float)
        length = float(np_norm(direction))

        if length < 1e-10:
            return

        style = self.arrow_style
        shaft_radius = shaft_radius or style.shaft_radius
        shaft_length = length * style.shaft_ratio
        tip_length = length * (1 - style.shaft_ratio)
        tip_radius = shaft_radius * style.tip_radius_factor

        direction_norm = direction / length
        shaft_end = start + direction_norm * shaft_length

        shaft = pv.Cylinder(
            center=(start + shaft_end) / 2,
            direction=direction_norm,
            radius=shaft_radius,
            height=shaft_length,
            resolution=style.resolution,
            capping=True,
        )
        self.plotter.add_mesh(shaft, color=color, smooth_shading=True)

        tip = pv.Cone(
            center=shaft_end + direction_norm * (tip_length / 2),
            direction=direction_norm,
            height=tip_length,
            radius=tip_radius,
            resolution=style.resolution,
            capping=True,
        )
        self.plotter.add_mesh(tip, color=color, smooth_shading=True)

    def arrow(
        self,
        start,
        direction,
        color: Color | None = None,
        shaft_radius: float | None = None,
    ):
        """
        Add an arrow to the scene.

        Args:
            start: Arrow origin point [x, y, z]
            direction: Arrow direction vector [dx, dy, dz]
            color: RGB tuple (0-1). If None, uses next palette color.
            shaft_radius: Override default shaft thickness
        """
        color = self._resolve_color(color)
        self._add_arrow_geometry(
            array(start),
            array(direction),
            color,
            shaft_radius,
        )

    def arrows(
        self,
        starts,
        directions,
        color: Color | None = None,
        colors: Palette | list | None = None,
        shaft_radius: float | None = None,
    ):
        """
        Add multiple arrows to the scene.

        Args:
            starts: Array of start points, shape (n, 3)
            directions: Array of direction vectors, shape (n, 3)
            color: Single color for all arrows
            colors: List or Palette of colors (one per arrow, cycles if needed)
            shaft_radius: Override default shaft thickness

        If neither color nor colors is provided, all arrows use the next
        single palette color (they share one color).
        """
        starts = array(starts)
        directions = array(directions)
        n = len(starts)

        if colors is not None:
            if isinstance(colors, Palette):
                color_list = colors.cycle(n)
            else:
                color_list = [colors[k % len(colors)] for k in range(n)]
        elif color is not None:
            color_list = [color] * n
        else:
            single_color = self._next_color()
            color_list = [single_color] * n

        for k in range(n):
            self._add_arrow_geometry(starts[k], directions[k], color_list[k], shaft_radius)

    # -------------------------------------------------------------------------
    # Curve Primitive
    # -------------------------------------------------------------------------

    def curve(
        self,
        points,
        color: Color | None = None,
        radius: float | None = None,
        closed: bool = False,
    ):
        """
        Add a smooth curve through the given points.

        Args:
            points: Array of points, shape (n, 3)
            color: RGB tuple (0-1). If None, uses next palette color.
            radius: Tube radius. If None, uses default.
            closed: Whether to close the curve into a loop.
        """
        import pyvista as pv

        color = self._resolve_color(color)
        style = self.curve_style
        radius = radius or style.radius

        points = array(points)
        n_spline = len(points) * style.spline_factor

        if closed:
            spline = pv.Spline(points, n_points=n_spline)
            spline = spline.compute_arc_length()
        else:
            spline = pv.Spline(points, n_points=n_spline)

        tube = spline.tube(radius=radius, n_sides=style.resolution)
        self.plotter.add_mesh(tube, color=color, smooth_shading=True)

    def curves(
        self,
        point_sets,
        color: Color | None = None,
        colors: Palette | list | None = None,
        radius: float | None = None,
    ):
        """
        Add multiple curves to the scene.

        Args:
            point_sets: List of point arrays, each shape (n_k, 3)
            color: Single color for all curves
            colors: List or Palette of colors
            radius: Tube radius
        """
        n = len(point_sets)

        if colors is not None:
            if isinstance(colors, Palette):
                color_list = colors.cycle(n)
            else:
                color_list = [colors[k % len(colors)] for k in range(n)]
        elif color is not None:
            color_list = [color] * n
        else:
            single_color = self._next_color()
            color_list = [single_color] * n

        for k, pts in enumerate(point_sets):
            self.curve(pts, color=color_list[k], radius=radius)

    # -------------------------------------------------------------------------
    # Point Primitive
    # -------------------------------------------------------------------------

    def point(
        self,
        position,
        color: Color | None = None,
        radius: float = 0.03,
    ):
        """
        Add a point (sphere) to the scene.

        Args:
            position: Point location [x, y, z]
            color: RGB tuple (0-1). If None, uses next palette color.
            radius: Sphere radius
        """
        import pyvista as pv

        color = self._resolve_color(color)
        sphere = pv.Sphere(radius=radius, center=position)
        self.plotter.add_mesh(sphere, color=color, smooth_shading=True)

    def points(
        self,
        positions,
        color: Color | None = None,
        colors: Palette | list | None = None,
        radius: float = 0.03,
    ):
        """
        Add multiple points to the scene.

        Args:
            positions: Array of positions, shape (n, 3)
            color: Single color for all points
            colors: List or Palette of colors
            radius: Sphere radius
        """
        positions = array(positions)
        n = len(positions)

        if colors is not None:
            if isinstance(colors, Palette):
                color_list = colors.cycle(n)
            else:
                color_list = [colors[k % len(colors)] for k in range(n)]
        elif color is not None:
            color_list = [color] * n
        else:
            single_color = self._next_color()
            color_list = [single_color] * n

        for k in range(n):
            self.point(positions[k], color=color_list[k], radius=radius)

    # -------------------------------------------------------------------------
    # Plane Primitive
    # -------------------------------------------------------------------------

    def plane(
        self,
        center,
        normal,
        size: float = 1.0,
        color: Color | None = None,
        opacity: float = 0.3,
    ):
        """
        Add a semi-transparent plane to the scene.

        Args:
            center: Center point of the plane
            normal: Normal vector to the plane
            size: Side length of the displayed square
            color: RGB tuple (0-1). If None, uses next palette color.
            opacity: Transparency (0=invisible, 1=opaque)
        """
        import pyvista as pv

        color = self._resolve_color(color)
        center = array(center, dtype=float)
        normal = array(normal, dtype=float)
        normal = normal / np_norm(normal)

        # Construct orthonormal basis for the plane
        if abs(normal[0]) < 0.9:
            u = cross(normal, array([1.0, 0.0, 0.0]))
        else:
            u = cross(normal, array([0.0, 1.0, 0.0]))
        u = u / np_norm(u)
        v = cross(normal, u)

        half = size / 2
        corners = array([
            center - half * u - half * v,
            center + half * u - half * v,
            center + half * u + half * v,
            center - half * u + half * v,
        ])

        plane_mesh = pv.Quadrilateral(corners)
        self.plotter.add_mesh(
            plane_mesh,
            color=color,
            opacity=opacity,
            smooth_shading=True,
        )

    # -------------------------------------------------------------------------
    # Display Control
    # -------------------------------------------------------------------------

    def reset_colors(self):
        """Reset color cycling to start of palette."""
        self._color_index = 0

    def camera(
        self,
        position: tuple[float, float, float] | None = None,
        focal_point: tuple[float, float, float] | None = None,
        view_up: tuple[float, float, float] | None = None,
    ):
        """
        Set camera position and orientation.

        Args:
            position: Camera location in world coordinates
            focal_point: Point the camera looks at
            view_up: Direction that appears "up" in the view
        """
        if position is not None:
            self.plotter.camera.position = position
        if focal_point is not None:
            self.plotter.camera.focal_point = focal_point
        if view_up is not None:
            self.plotter.camera.up = view_up

    def show(self, block: bool = True):
        """
        Display the visualization.

        Args:
            block: If True, blocks until window is closed. If False, returns
                   immediately (useful for multiple windows or animations).
        """
        if self._title:
            self.plotter.add_title(
                self._title,
                font_size=12,
                color=self.theme.label,
            )

        if block:
            self.plotter.show()
        else:
            self.plotter.show(interactive_update=True, auto_close=False)

    def screenshot(self, filename: str, scale: int = 2):
        """
        Save the current view to an image file.

        Args:
            filename: Output file path (supports png, jpg, etc.)
            scale: Resolution multiplier for higher quality
        """
        self.plotter.screenshot(filename, scale=scale)

    def close(self):
        """Close the plotter window."""
        self.plotter.close()
