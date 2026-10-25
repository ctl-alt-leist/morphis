"""
Sketch

A pen-and-ink figure. Marks are recorded in 3D drawing coordinates as they are
added and rendered to the page only on save, once the whole figure is known,
so the layout, the stroke scale, and the depth ordering are all settled from
the complete scene.

Geometry passed as a morphis Vector is true-space geometry, carried into the
drawing through the sketch's projection: a View (Orthographic, Perspective,
Stereographic) resolved against the vector's metric, or a given Projection such
as Projection.linear, the deliberately unfaithful map that sends each true
direction to a chosen drawing direction. Geometry passed as a plain triple is
a drawing-space placement: where the figure chooses to put something.

Points (positions, curves, surfaces, labels) may go through any projection.
Marks that also take displacements (a vector's arrow, a plane's spanning
vectors, a circle's plane) set them out from a drawing-space placement, which
needs an affine projection, one whose displacements have images of their own.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import withStroke
from numpy import (
    abs as np_abs,
    add,
    arange,
    arctan2,
    asarray,
    ceil,
    clip as clamp,
    concatenate,
    cos,
    cross,
    cumsum,
    einsum,
    exp as exponential,
    floor,
    full,
    inf,
    interp,
    linspace,
    meshgrid,
    minimum,
    pi,
    repeat,
    rint,
    sin,
    sqrt,
    stack,
    where,
    zeros,
)
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter

from morphis.elements.vector import Vector
from morphis.visuals.ink.camera import Camera
from morphis.visuals.ink.fonts import font_settings
from morphis.visuals.ink.space import OrganicSpace
from morphis.visuals.ink.theme import RGB, InkTheme, get_ink_theme
from morphis.visuals.projective import Orthographic, Projection, View, resolve


Point = Vector | tuple[float, float, float] | NDArray

# Draw order of mark kinds; within a layer, farther marks are drawn first
LAYERS = {
    "space": 1,
    "shell": 2,
    "surface": 2,
    "plane": 3,
    "construction": 5,
    "curve": 5,
    "vector": 6,
    "point": 7,
    "label": 9,
}

# Stroke weights in points
WEIGHTS = {"hair": 0.45, "fine": 0.8, "regular": 1.25, "bold": 1.9, "heavy": 2.6}


@dataclass
class Mark:
    """One recorded drawing instruction."""

    kind: str
    points: NDArray  # drawing coordinates, shape (n, 3)
    style: dict[str, Any] = field(default_factory=dict)


class Sketch:
    """
    A pen-and-ink conceptual figure.

    Args:
        camera: Viewpoint onto the drawing space
        theme: Ink theme name or InkTheme ("ink", "parchment", "chalkboard")
        projection: View or Projection from the true space into the drawing space;
            orthographic onto the directions 1, 2, 3 by default
        size: Figure size in inches
        seed: Seed for stipple and pen wobble, so renders are repeatable
    """

    def __init__(
        self,
        camera: Camera | None = None,
        theme: str | InkTheme = "ink",
        projection: View | Projection | None = None,
        size: tuple[float, float] = (8.0, 6.0),
        seed: int = 0,
    ):
        self.camera = camera or Camera()
        self.theme = get_ink_theme(theme)
        self.projection = Orthographic() if projection is None else projection
        self.size = size
        self.seed = seed
        self.marks: list[Mark] = []
        self.spaces: list[tuple[OrganicSpace, dict]] = []

    # =========================================================================
    # Coordinates
    # =========================================================================

    def drawn(self, point: Point) -> NDArray:
        """Drawing coordinates of a true-space point (a Vector of any lot) or a drawing-space triple."""
        is_true = isinstance(point, Vector)
        coordinates = resolve(self.projection, point.metric)(point) if is_true else asarray(point, dtype=float)

        return coordinates

    def carried(self, displacement: Point) -> NDArray:
        """
        Drawing displacement of a true-space Vector, or a drawing-space triple as given.

        Raises:
            ValueError: if the projection is not affine, since a central
                projection draws a displacement differently at every point.
        """
        is_true = isinstance(displacement, Vector)
        if is_true:
            projection = resolve(self.projection, displacement.metric)
            if not projection.is_affine:
                raise ValueError(f"{projection!r} is not affine, so a displacement has no drawing of its own")

            result = projection(displacement) - projection(zeros(displacement.dim))

        else:
            result = asarray(displacement, dtype=float)

        return result

    def _ink(self, color: str | RGB | None, level: float | None = None) -> RGB:
        """Resolve a color name, RGB triple, or graphite level."""
        is_named = isinstance(color, str)
        resolved = (
            self.theme.color(color) if is_named else color if color is not None else self.theme.gray(level or 1.0)
        )

        return resolved

    # =========================================================================
    # Marks
    # =========================================================================

    def space(self, space: OrganicSpace, weight: str = "bold", stipple: float = 1.0, light: float = 210.0) -> None:
        """
        Draw an enclosing organic space: a pen silhouette with rim stipple.

        Args:
            space: The organic space
            weight: Silhouette stroke weight
            stipple: Stipple density multiplier; 0 for a bare outline
            light: Page angle in degrees of the side that falls in shadow
        """
        self.spaces.append((space, {"weight": weight, "stipple": stipple, "light": light}))

    def plane(
        self,
        u: Point,
        v: Point,
        at: Point = (0.0, 0.0, 0.0),
        span: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
        grid: int = 4,
        tone: float = 0.18,
        color: str | RGB | None = None,
    ) -> None:
        """
        Draw a plane patch spanned by u and v, as a stippled parallelogram with a faint grid.

        Args:
            u, v: Spanning vectors (true-space Vectors are depicted)
            at: Drawing-space origin of the patch
            span: Parameter ranges along u and along v
            grid: Number of grid cells along each side; 0 for none
            tone: Stipple density of the face
            color: Edge ink
        """
        origin, du, dv = self.drawn(at), self.carried(u), self.carried(v)
        (s0, s1), (t0, t1) = span
        corners = asarray([origin + s * du + t * dv for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))])
        style = {"grid": grid, "tone": tone, "color": color, "origin": origin, "du": du, "dv": dv, "span": span}
        self.marks.append(Mark("plane", corners, style))

    def vector(
        self,
        v: Point,
        at: Point = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        weight: str = "regular",
        color: str | RGB | None = None,
        head: float = 1.0,
        label: str | None = None,
        label_offset: float = 12.0,
        label_along: float = 0.5,
        label_size: float = 15.0,
    ) -> NDArray:
        """
        Draw an arrow for v from the drawing-space point at.

        Args:
            label: Text set beside the shaft rather than at the tip, clear of lines leaving the tip
            label_offset: Distance from the shaft in points; positive is left of the arrow's direction
            label_along: Where along the shaft the label sits, 0 at the tail and 1 at the tip

        Returns:
            Drawing coordinates of the tip, for attaching labels and construction lines
        """
        tail = self.drawn(at)
        tip = tail + scale * self.carried(v)
        style = {
            "weight": weight,
            "color": color,
            "head": head,
            "label": label,
            "label_offset": label_offset,
            "label_along": label_along,
            "label_size": label_size,
        }
        self.marks.append(Mark("vector", stack([tail, tip]), style))

        return tip

    def line(
        self,
        start: Point,
        end: Point,
        dashed: bool = True,
        weight: str = "fine",
        color: str | RGB | None = None,
    ) -> None:
        """Draw a construction line, dashed by default."""
        points = stack([self.drawn(start), self.drawn(end)])
        self.marks.append(Mark("construction", points, {"dashed": dashed, "weight": weight, "color": color}))

    def circle(
        self,
        center: Point,
        u: Point,
        v: Point,
        radius: float = 1.0,
        arc: tuple[float, float] = (0.0, 2 * pi),
        arrow: bool = False,
        weight: str = "fine",
        level: float = 0.6,
    ) -> None:
        """Draw a circle (or arc) of the given radius in the plane of unit directions u and v."""
        du, dv = self.carried(u), self.carried(v)
        du, dv = du / norm(du), dv / norm(dv)
        angles = linspace(arc[0], arc[1], 160)
        points = self.drawn(center) + radius * (cos(angles)[:, None] * du + sin(angles)[:, None] * dv)
        style = {"weight": weight, "level": level, "arrow": arrow, "dashed": False}
        self.marks.append(Mark("curve", points, style))

    def curve(
        self,
        points: Vector | list[Point] | NDArray,
        dashed: bool = False,
        arrow: bool = False,
        weight: str = "fine",
        level: float = 0.6,
    ) -> None:
        """Draw a polyline through points: a true-space Vector with a lot, a list of points, or drawing triples."""
        is_list = isinstance(points, list)
        drawn = asarray([self.drawn(p) for p in points]) if is_list else self.drawn(points).reshape(-1, 3)
        style = {"weight": weight, "level": level, "arrow": arrow, "dashed": dashed}
        self.marks.append(Mark("curve", drawn, style))

    def point(self, p: Point, radius: float = 3.2, color: str | RGB | None = None) -> NDArray:
        """Draw a solid dot; returns its drawing coordinates."""
        position = self.drawn(p)
        self.marks.append(Mark("point", position[None, :], {"radius": radius, "color": color}))

        return position

    def shell(
        self,
        points: Vector | NDArray,
        cells: NDArray,
        weight: str = "regular",
        stipple: float = 1.0,
        light: tuple[float, float, float] = (-0.45, 0.6, 0.65),
        far_level: float = 0.4,
        veil: float = 0.3,
        curves: list[Vector | NDArray] | None = None,
        curve_weight: str = "fine",
        dots: Vector | NDArray | None = None,
        dot_radius: float = 4.0,
        labels: list[str | None] | None = None,
        label_offset: tuple[float, float] = (9.0, 7.0),
        label_size: float = 15.0,
    ) -> None:
        """
        Draw a translucent surface in pen and stipple: its contour inked, its body shaded by stipple.

        The contour is where the surface turns from facing the eye to facing
        away, found between vertices from interpolated normals and inked with a
        pen wobble, lighter the farther it is. Stipple is densest where the
        surface turns edge-on to the eye and heavier on the side away from the
        light. No grid is drawn.

        The shell is translucent, not transparent: anything a nearer sheet of
        it covers, whether its own far contour and stipple or the curves and
        dots given with it, is drawn at the strength veil. Covering is decided
        against a depth buffer of the shell rasterized on the page.

        Args:
            points: Vertex positions, true-space Vectors (projected) or drawing triples, shape (n, 3)
            cells: Vertex indices per cell, squares in binary corner order (00, 10, 01, 11) or triangles
            weight: Stroke weight of the nearest contour
            stipple: Stipple density multiplier; 0 for the contour alone
            light: Light direction in the camera frame (right, up, toward the eye)
            far_level: Graphite level of the farthest marks in front of the shell
            veil: Strength, from 0 to 1, of whatever a nearer sheet of the shell covers
            curves: Polylines drawn with the shell and veiled where it covers them,
                true-space Vectors (projected) or drawing points
            curve_weight: Stroke weight of the curves
            dots: Points drawn with the shell and veiled where it covers them
            dot_radius: Radius of the dots in points
            labels: Text beside each dot, veiled with it, or None for none
            label_offset: Offset of the labels from their dots, in points
            label_size: Font size of the labels
        """
        drawn = self.drawn(points) if isinstance(points, Vector) else asarray(points, dtype=float)
        cells = asarray(cells, dtype=int)
        triangles = concatenate([cells[:, [0, 1, 3]], cells[:, [0, 3, 2]]]) if cells.shape[1] == 4 else cells
        style = {
            "triangles": triangles,
            "weight": weight,
            "stipple": stipple,
            "light": light,
            "far_level": far_level,
            "veil": veil,
            "curves": [self.drawn(curve).reshape(-1, 3) for curve in curves or []],
            "curve_weight": curve_weight,
            "dots": self.drawn(dots).reshape(-1, 3) if dots is not None else zeros((0, 3)),
            "dot_radius": dot_radius,
            "labels": labels or [],
            "label_offset": label_offset,
            "label_size": label_size,
        }
        self.marks.append(Mark("shell", drawn.reshape(-1, 3), style))

    def surface(
        self,
        points: Vector | NDArray,
        cells: NDArray,
        colors: str | RGB | list | None = None,
        edges: str | None = "hair",
        edge_level: float = 0.75,
        light: tuple[float, float, float] = (-0.45, 0.6, 0.65),
        dots: Vector | NDArray | None = None,
        dot_radius: float = 3.0,
        dot_color: str | RGB | None = "ink",
    ) -> None:
        """
        Draw a shaded surface, its cells painted far to near so nearer cells hide farther ones.

        Dots on the surface are painted in the same order, so a dot on a sheet
        behind another is hidden by it.

        Args:
            points: Vertex positions, true-space Vectors (depicted) or drawing triples, shape (n, 3)
            cells: Vertex indices per cell, squares in binary corner order (00, 10, 01, 11) or triangles
            colors: One color for all cells, or one per cell; graphite by default
            edges: Stroke weight of the cell edges, which draw the coordinate grid, or None
            edge_level: Graphite level of the edges
            light: Light direction in the camera frame (right, up, toward the eye)
            dots: Points on the surface to mark, shape (n, 3)
            dot_radius: Radius of the dots in points
            dot_color: Color of the dots
        """
        drawn = self.drawn(points) if isinstance(points, Vector) else asarray(points, dtype=float)
        cells = asarray(cells, dtype=int)
        polygons = cells[:, [0, 1, 3, 2]] if cells.shape[1] == 4 else cells
        marked = (
            (self.drawn(dots) if isinstance(dots, Vector) else asarray(dots, dtype=float)).reshape(-1, 3)
            if dots is not None
            else zeros((0, 3))
        )
        style = {
            "cells": polygons,
            "colors": colors,
            "edges": edges,
            "edge_level": edge_level,
            "light": light,
            "dots": marked,
            "dot_radius": dot_radius,
            "dot_color": dot_color,
        }
        self.marks.append(Mark("surface", drawn.reshape(-1, 3), style))

    def label(
        self,
        text: str,
        at: Point,
        offset: tuple[float, float] = (6.0, 4.0),
        size: float = 13.0,
        color: str | RGB | None = None,
    ) -> None:
        """Place text beside a drawing-space point, offset in points."""
        position = self.drawn(at)
        self.marks.append(
            Mark("label", position[None, :], {"text": text, "offset": offset, "size": size, "color": color})
        )

    # =========================================================================
    # Rendering
    # =========================================================================

    def page_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Page rectangle that holds the whole sketch with a margin, fitted to the figure aspect.

        Returns:
            lower, upper: Opposite corners of the page in focal-plane units
        """
        outlines = [space.outline(self.camera)[0] for space, _ in self.spaces]
        page_points = outlines + [self.camera.project(m.points)[0] for m in self.marks]
        everything = concatenate(page_points) if page_points else asarray([[0.0, 0.0]])
        lower, upper = everything.min(axis=0), everything.max(axis=0)
        margin = 0.06 * (upper - lower).max()
        lower, upper = lower - margin, upper + margin

        # One unit is equal in x and y
        size = asarray(self.size)
        scale = ((upper - lower) / size).max()
        middle = 0.5 * (lower + upper)
        half = 0.5 * scale * size

        return middle - half, middle + half

    def render(self, dpi: int = 200, bounds: tuple[NDArray, NDArray] | None = None) -> Figure:
        """
        Render the sketch to a matplotlib Figure.

        Args:
            dpi: Resolution
            bounds: Fixed page rectangle (from page_bounds); holding it fixed keeps animation frames registered
        """
        figure = Figure(figsize=self.size, dpi=dpi, facecolor=self.theme.paper)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        axes.set_facecolor(self.theme.paper)

        lower, upper = bounds if bounds is not None else self.page_bounds()
        axes.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        axes.set_aspect("equal")
        units_per_point = (upper[0] - lower[0]) / self.size[0] / 72.0

        # Each space and mark draws from its own seeded stream, keyed by the order it was added,
        # so its stipple and pen wobble do not change when other marks move or reorder
        for count, (space, options) in enumerate(self.spaces):
            outline, center = space.outline(self.camera)
            rng = default_rng([self.seed, 0, count])
            self._render_space(axes, outline, center, options, units_per_point, rng)

        depths = [self.camera.project(m.points)[1].mean() for m in self.marks]
        entries = sorted(
            zip(range(len(self.marks)), self.marks, depths, strict=True),
            key=lambda entry: (LAYERS[entry[1].kind], -entry[2]),
        )
        for order, (count, mark, _) in enumerate(entries):
            zorder = LAYERS[mark.kind] + order * 1e-4
            page, _ = self.camera.project(mark.points)
            renderer = getattr(self, f"_render_{mark.kind}")
            renderer(axes, page, mark, zorder, units_per_point, default_rng([self.seed, 1, count]))

        return figure

    def save(self, path: str | Path, dpi: int = 200, bounds: tuple[NDArray, NDArray] | None = None) -> Path:
        """Render and write the sketch; the format follows the extension (png, svg, pdf)."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with rc_context(font_settings(self.theme.font)):
            figure = self.render(dpi=dpi, bounds=bounds)
            figure.savefig(target, dpi=dpi, facecolor=self.theme.paper)

        return target

    def frame(self, dpi: int = 150, bounds: tuple[NDArray, NDArray] | None = None) -> NDArray:
        """Render to an RGB image array of shape (height, width, 3), for assembling animations."""
        with rc_context(font_settings(self.theme.font)):
            figure = self.render(dpi=dpi, bounds=bounds)
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            image = asarray(canvas.buffer_rgba())[..., :3].copy()

        return image

    # -------------------------------------------------------------------------
    # Mark renderers
    # -------------------------------------------------------------------------

    def _render_space(self, axes, outline, center, options, units_per_point, rng) -> None:
        offset = outline - center
        reach = norm(offset, axis=-1)
        angle = arctan2(offset[:, 1], offset[:, 0])
        shadow_angle = options["light"] * pi / 180.0
        shadow = 0.5 * (1.0 + cos(angle - shadow_angle))

        # Silhouette: heavier on the shadow side, with a faint pen wobble
        base = WEIGHTS[options["weight"]]
        stroke = _wobble(outline, 0.8 * units_per_point, rng)
        widths = base * (0.8 + 0.55 * shadow[:-1])
        segments = stack([stroke[:-1], stroke[1:]], axis=1)
        axes.add_collection(
            LineCollection(segments, linewidths=widths, colors=[self.theme.ink], capstyle="round", zorder=2)
        )

        if options["stipple"] > 0:
            # Rim stipple: a dense band just inside the silhouette, thinning inward, darker on the shadow side
            spacing = 1.25 * units_per_point
            lower, upper = outline.min(axis=0), outline.max(axis=0)
            gx, gy = meshgrid(arange(lower[0], upper[0], spacing), arange(lower[1], upper[1], spacing))
            dots = stack([gx.ravel(), gy.ravel()], axis=-1)
            dots = dots + rng.uniform(-0.5, 0.5, size=dots.shape) * spacing
            relative = dots - center
            dot_angle = arctan2(relative[:, 1], relative[:, 0])
            boundary = interp(dot_angle, angle[:-1], reach[:-1], period=2 * pi)
            depth_in = 1.0 - norm(relative, axis=-1) / boundary
            dot_shadow = 0.5 * (1.0 + cos(dot_angle - shadow_angle))
            band = 0.03 + 0.045 * dot_shadow
            rim = exponential(-((clamp(depth_in, 0.0, None) / band) ** 0.8))
            density = options["stipple"] * (0.008 + 0.9 * rim * (0.3 + 0.7 * dot_shadow))
            keep = (depth_in > 0.002) & (rng.uniform(size=len(dots)) < clamp(density, 0.0, 1.0))
            sizes = rng.uniform(0.06, 0.32, size=int(keep.sum()))
            axes.scatter(*dots[keep].T, s=sizes, c=[self.theme.ink], linewidths=0, zorder=1)

    def _render_shell(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        points, triangles = mark.points, style["triangles"]
        _, depth = self.camera.project(points)
        near, far = depth.min(), depth.max()
        buffer = _DepthBuffer(page, depth, triangles)

        # Graphite level: lighter with depth in front, and veiled where a nearer sheet covers
        def level(at_page: NDArray, at_depth: NDArray) -> NDArray:
            nearness = clamp((far - at_depth) / max(far - near, 1e-12), 0.0, 1.0)
            shade = style["far_level"] + (1.0 - style["far_level"]) * nearness
            return where(buffer.covers(at_page, at_depth), style["veil"] * shade, shade)

        def ink(at_page: NDArray, at_depth: NDArray, weight: str, wobble: float, z: float) -> None:
            # The wobbled stroke is resampled evenly by arc length; depth follows it the same way
            stroke = _wobble(at_page, wobble * units_per_point, rng, samples=4 * len(at_page))
            arc = concatenate([[0.0], cumsum(norm(at_page[1:] - at_page[:-1], axis=-1))])
            stroke_depth = interp(linspace(0.0, arc[-1], len(stroke)), arc, at_depth)
            shade = level(stroke, stroke_depth)[:-1]
            segments = stack([stroke[:-1], stroke[1:]], axis=1)
            axes.add_collection(
                LineCollection(
                    segments,
                    linewidths=WEIGHTS[weight] * (0.55 + 0.45 * shade),
                    colors=[self.theme.gray(value) for value in shade],
                    capstyle="round",
                    zorder=z,
                )
            )

        # Facing: the normal against the direction toward the eye, per face and per vertex
        corners = points[triangles]
        normals = cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        vertex_normals = zeros(points.shape)
        for corner in range(3):
            add.at(vertex_normals, triangles[:, corner], normals)

        toward = self._toward_eye(corners.mean(axis=1))
        unit = normals / clamp(norm(normals, axis=-1, keepdims=True), 1e-12, None)
        facing = einsum("na, na -> n", unit, toward)
        vertex_facing = einsum("na, na -> n", vertex_normals, self._toward_eye(points))

        # Contour: where the vertex facing changes sign
        for contour in _contours(points, triangles, vertex_facing):
            ink(*self.camera.project(contour), style["weight"], 0.6, zorder + 1e-5)

        if style["stipple"] > 0:
            # Stipple: densest where the surface turns edge-on, heavier away from the light
            light = einsum("ab, a -> b", self.camera.frame, asarray(style["light"], dtype=float))
            light = light / norm(light)
            front = unit * where(facing >= 0, 1.0, -1.0)[:, None]
            shadow = 0.5 * (1.0 - front @ light)
            rim = exponential(-((np_abs(facing) / 0.28) ** 0.9))
            density = style["stipple"] * (0.012 + 0.95 * rim * (0.3 + 0.7 * shadow))

            page_corners, depth_corners = page[triangles], depth[triangles]
            first_side, second_side = page_corners[:, 1] - page_corners[:, 0], page_corners[:, 2] - page_corners[:, 0]
            spans = first_side[:, 0] * second_side[:, 1] - first_side[:, 1] * second_side[:, 0]
            spacing = 1.25 * units_per_point
            counts = rng.poisson(0.5 * np_abs(spans) / spacing**2 * clamp(density, 0.0, 1.0))
            owner = repeat(arange(len(triangles)), counts)
            first, second = rng.uniform(size=(2, len(owner)))
            is_folded = first + second > 1.0
            first, second = where(is_folded, 1.0 - first, first), where(is_folded, 1.0 - second, second)

            def at(values: NDArray) -> NDArray:
                base = values[owner, 0]
                weights = (first, second) if values.ndim == 2 else (first[:, None], second[:, None])
                return base + weights[0] * (values[owner, 1] - base) + weights[1] * (values[owner, 2] - base)

            dots, dot_depth = at(page_corners), at(depth_corners)

            # A covered dot is also less likely to be drawn, so the far sheets thin out behind the near ones
            shade = level(dots, dot_depth)
            is_kept = rng.uniform(size=len(owner)) < where(buffer.covers(dots, dot_depth), 0.5, 1.0)
            sizes = rng.uniform(0.08, 0.36, size=len(owner)) * (0.5 + 0.5 * shade)
            colors = [self.theme.gray(value) for value in shade[is_kept]]
            axes.scatter(*dots[is_kept].T, s=sizes[is_kept], c=colors, linewidths=0, zorder=zorder)

        # Curves and dots given with the shell, veiled where it covers them
        for curve in style["curves"]:
            ink(*self.camera.project(curve), style["curve_weight"], 0.0, zorder + 2e-5)

        if len(style["dots"]):
            dot_page, dot_depth = self.camera.project(style["dots"])
            shade = level(dot_page, dot_depth)
            axes.scatter(
                *dot_page.T,
                s=style["dot_radius"] ** 2 * 3.2,
                c=[self.theme.gray(value) for value in shade],
                linewidths=0,
                zorder=zorder + 3e-5,
            )
            for text, position, value in zip(style["labels"], dot_page, shade, strict=False):
                if text is not None:
                    color = self.theme.gray(value)
                    self._annotate(
                        axes, text, position, style["label_offset"], style["label_size"], color, zorder + 4e-5
                    )

    def _toward_eye(self, points: NDArray) -> NDArray:
        """Unit directions from points toward the eye; for an orthographic camera, the view direction."""
        if self.camera.fov > 0:
            eye = asarray(self.camera.focal_point) + self.camera.distance * self.camera.view_direction
            toward = eye - points
            result = toward / norm(toward, axis=-1, keepdims=True)

        else:
            result = zeros(points.shape) + self.camera.view_direction

        return result

    def _render_surface(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        polygons = style["cells"]
        _, depth = self.camera.project(mark.points)
        cell_depth = depth[polygons].mean(axis=-1)
        order = (-cell_depth).argsort()
        polygons, cell_depth = polygons[order], cell_depth[order]

        # Two-sided Lambert shading from the cell normal in drawing space
        corners = mark.points[polygons]
        span_1 = corners[:, 2] - corners[:, 0]
        span_2 = corners[:, -1] - corners[:, 1] if polygons.shape[1] == 4 else corners[:, 1] - corners[:, 0]
        normals = cross(span_1, span_2)
        normals = normals / clamp(norm(normals, axis=-1, keepdims=True), 1e-12, None)
        light = einsum("ab, a -> b", self.camera.frame, asarray(style["light"], dtype=float))
        light = light / norm(light)
        shade = abs(normals @ light)

        colors = style["colors"]
        is_per_cell = isinstance(colors, (list, tuple)) and len(colors) == len(style["cells"]) and len(colors) != 3
        base = (
            asarray([self._ink(c) for c in colors])[order]
            if is_per_cell
            else asarray(self._ink(colors, 0.3) if colors is not None else self.theme.gray(0.3))[None, :]
        )
        paper = asarray(self.theme.paper)
        brightness = (0.2 + 0.8 * shade)[:, None]
        faces = paper + (base - paper) * (1.15 - 0.75 * brightness)
        faces = clamp(faces, 0.0, 1.0)

        # Dots sit just in front of the sheet they mark and split the far-to-near cell order
        dot_page, dot_depth = self.camera.project(style["dots"])
        nudge = 1e-3 * (depth.max() - depth.min())
        dot_order = (-dot_depth).argsort()
        splits = [int((cell_depth > dot_depth[n] - nudge).sum()) for n in dot_order]
        bounds = [0, *splits, len(polygons)]

        edges = style["edges"]
        step = 1e-6
        for count, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            if stop > start:
                collection = PolyCollection(
                    page[polygons[start:stop]],
                    facecolors=faces[start:stop],
                    edgecolors=[self.theme.gray(style["edge_level"])] if edges else "face",
                    linewidths=WEIGHTS[edges] if edges else 0.25,
                    zorder=zorder + 2 * count * step,
                )
                axes.add_collection(collection)

            if count < len(dot_order):
                axes.scatter(
                    *dot_page[dot_order[count]],
                    s=style["dot_radius"] ** 2 * 3.2,
                    c=[self._ink(style["dot_color"])],
                    linewidths=0,
                    zorder=zorder + (2 * count + 1) * step,
                )

    def _render_plane(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        ink = self._ink(style["color"])
        axes.add_patch(Polygon(page, closed=True, facecolor=self.theme.paper, edgecolor="none", zorder=zorder))

        # Face stipple, slightly heavier toward the edges
        path = MplPath(page)
        lower, upper = page.min(axis=0), page.max(axis=0)
        spacing = 1.7 * units_per_point
        gx, gy = meshgrid(arange(lower[0], upper[0], spacing), arange(lower[1], upper[1], spacing))
        dots = stack([gx.ravel(), gy.ravel()], axis=-1) + rng.uniform(-0.5, 0.5, size=(gx.size, 2)) * spacing
        inside = path.contains_points(dots)
        edge_distance = _distance_to_polygon(dots, page)
        reach = 0.08 * (upper - lower).min()
        density = style["tone"] * (0.45 + 0.9 * exponential(-edge_distance / reach))
        keep = inside & (rng.uniform(size=len(dots)) < density)
        axes.scatter(
            *dots[keep].T,
            s=rng.uniform(0.08, 0.35, int(keep.sum())),
            c=[self.theme.gray(0.7)],
            linewidths=0,
            zorder=zorder + 1e-5,
        )

        # Faint dashed grid
        if style["grid"] > 0:
            (s0, s1), (t0, t1) = style["span"]
            origin, du, dv = style["origin"], style["du"], style["dv"]
            rules = []
            for s in linspace(s0, s1, style["grid"] + 1)[1:-1]:
                rules.append([origin + s * du + t0 * dv, origin + s * du + t1 * dv])
            for t in linspace(t0, t1, style["grid"] + 1)[1:-1]:
                rules.append([origin + s0 * du + t * dv, origin + s1 * du + t * dv])
            rule_page = self.camera.project(asarray(rules))[0]
            axes.add_collection(
                LineCollection(
                    rule_page,
                    linewidths=WEIGHTS["hair"],
                    colors=[self.theme.gray(0.35)],
                    linestyles=(0, (3, 2.5)),
                    zorder=zorder + 2e-5,
                )
            )

        edge = _wobble(concatenate([page, page[:1]]), 0.3 * units_per_point, rng, samples=200)
        axes.plot(*edge.T, color=ink, linewidth=WEIGHTS["regular"], solid_joinstyle="round", zorder=zorder + 3e-5)

    def _render_construction(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        dash = (0, (4.5, 3.0)) if style["dashed"] else "solid"
        axes.plot(
            *page.T,
            color=self._ink(style["color"]),
            linewidth=WEIGHTS[style["weight"]],
            linestyle=dash,
            dash_capstyle="round",
            zorder=zorder,
        )

    def _render_curve(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        ink = self.theme.gray(style["level"])
        dash = (0, (1.0, 2.2)) if style["dashed"] else "solid"
        axes.plot(
            *page.T,
            color=ink,
            linewidth=WEIGHTS[style["weight"]],
            linestyle=dash,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=zorder,
        )
        if style["arrow"]:
            _arrowhead(axes, page[-2], page[-1], 6.0 * units_per_point, ink, zorder)

    def _render_vector(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        ink = self._ink(style["color"])
        width = WEIGHTS[style["weight"]]
        head_length = (7.0 + 2.6 * width) * style["head"] * units_per_point
        tail, tip = page
        direction = (tip - tail) / norm(tip - tail)
        shaft_end = tip - 0.7 * head_length * direction
        wobble = min(0.4 * units_per_point, 0.006 * norm(shaft_end - tail))
        shaft = _wobble(stack([tail, shaft_end]), wobble, rng, samples=40)
        axes.plot(*shaft.T, color=ink, linewidth=width, solid_capstyle="round", zorder=zorder)
        _arrowhead(axes, tail, tip, head_length, ink, zorder)

        if style["label"] is not None:
            beside = tail + style["label_along"] * (tip - tail)
            normal = asarray([-direction[1], direction[0]]) * style["label_offset"]
            self._annotate(axes, style["label"], beside, tuple(normal), style["label_size"], ink, LAYERS["label"])

    def _render_point(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        axes.scatter(*page.T, s=style["radius"] ** 2 * 3.2, c=[self._ink(style["color"])], linewidths=0, zorder=zorder)

    def _render_label(self, axes, page, mark, zorder, units_per_point, rng) -> None:
        style = mark.style
        self._annotate(axes, style["text"], page[0], style["offset"], style["size"], self._ink(style["color"]), zorder)

    def _annotate(
        self, axes, text: str, anchor: NDArray, offset: tuple[float, float], size: float, color: RGB, zorder: float
    ) -> None:
        """Set text centered at an offset in points from a page point, with a paper halo."""
        axes.annotate(
            text,
            xy=anchor,
            xytext=offset,
            textcoords="offset points",
            fontsize=size,
            color=color,
            ha="center",
            va="center",
            zorder=zorder,
            path_effects=[withStroke(linewidth=3.5, foreground=self.theme.paper)],
        )


# =============================================================================
# Pen geometry
# =============================================================================


def _wobble(polyline: NDArray, amplitude: float, rng, samples: int | None = None) -> NDArray:
    """Resample a polyline by arc length and displace it along its normal with smooth seeded noise."""
    steps = norm(polyline[1:] - polyline[:-1], axis=-1)
    arc = concatenate([[0.0], cumsum(steps)])
    count = samples or len(polyline)
    s = linspace(0.0, arc[-1], count)
    resampled = stack([interp(s, arc, polyline[:, 0]), interp(s, arc, polyline[:, 1])], axis=-1)

    tangent = concatenate([resampled[1:] - resampled[:-1], resampled[-1:] - resampled[-2:-1]])
    tangent = tangent / clamp(norm(tangent, axis=-1, keepdims=True), 1e-12, None)
    normal = stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    phase = s / max(arc[-1], 1e-12)
    noise = sum(rng.normal() / k * sin(2 * pi * k * phase + rng.uniform(0, 2 * pi)) for k in (1, 2, 3, 5))
    ends = sqrt(clamp(phase * (1 - phase) * 4, 0.0, 1.0))
    is_closed = norm(polyline[0] - polyline[-1]) < 1e-9
    envelope = 1.0 if is_closed else ends
    displaced = resampled + (amplitude * noise * envelope)[:, None] * normal

    return displaced


class _DepthBuffer:
    """
    The nearest depth of a triangle mesh at each pixel of a grid over its page extent.

    A query point is covered when the mesh around its page position is nearer
    than it by more than a small tolerance, so a point lying on the mesh is
    not covered by the sheet it lies on, even where that sheet is steep.
    """

    def __init__(self, page: NDArray, depth: NDArray, triangles: NDArray, resolution: int = 480):
        lower, upper = page.min(axis=0), page.max(axis=0)
        self.lower = lower
        self.scale = (resolution - 1) / float((upper - lower).max())
        self.tolerance = 0.03 * float(depth.max() - depth.min())
        width, height = (ceil((upper - lower) * self.scale) + 1).astype(int)
        self.depth = full((height, width), inf)

        pixels = (page - lower) * self.scale
        for a, b, c in triangles:
            self._raster(pixels[[a, b, c]], depth[[a, b, c]])

        # Covered only where the whole neighborhood is nearer, so a steep edge does not cover its own sheet
        self.depth = maximum_filter(self.depth, size=3)

    def _raster(self, corners: NDArray, depths: NDArray) -> None:
        low, high = floor(corners.min(axis=0)).astype(int), ceil(corners.max(axis=0)).astype(int)
        xs, ys = meshgrid(arange(low[0], high[0] + 1), arange(low[1], high[1] + 1))
        (x0, y0), (x1, y1), (x2, y2) = corners
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) > 1e-12:
            weight_1 = ((xs - x0) * (y2 - y0) - (x2 - x0) * (ys - y0)) / area
            weight_2 = ((x1 - x0) * (ys - y0) - (xs - x0) * (y1 - y0)) / area
            weight_0 = 1.0 - weight_1 - weight_2
            inside = (weight_0 >= -0.02) & (weight_1 >= -0.02) & (weight_2 >= -0.02)
            values = weight_0 * depths[0] + weight_1 * depths[1] + weight_2 * depths[2]
            region = self.depth[ys[inside], xs[inside]]
            self.depth[ys[inside], xs[inside]] = minimum(region, values[inside])

    def covers(self, page: NDArray, depth: NDArray) -> NDArray:
        """Whether a nearer part of the mesh lies in front of each page point at its depth."""
        pixels = rint((asarray(page) - self.lower) * self.scale).astype(int)
        height, width = self.depth.shape
        is_on = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        nearest = full(len(pixels), inf)
        nearest[is_on] = self.depth[pixels[is_on, 1], pixels[is_on, 0]]

        return nearest < asarray(depth) - self.tolerance


def _contours(points: NDArray, triangles: NDArray, values: NDArray) -> list[NDArray]:
    """
    Polylines along the zero set of a function given at the vertices of a triangle mesh.

    On each edge whose ends differ in sign the crossing is interpolated
    linearly; the crossings of a triangle are joined, and segments sharing an
    edge are chained, so a closed contour comes back as one closed polyline.
    """
    crossings = {}

    def crossing(a: int, b: int) -> tuple[int, int]:
        key = (min(a, b), max(a, b))
        if key not in crossings:
            low, high = key
            share = values[low] / (values[low] - values[high])
            crossings[key] = points[low] + share * (points[high] - points[low])

        return key

    segments = []
    for a, b, c in triangles:
        keys = [crossing(i, j) for i, j in ((a, b), (b, c), (c, a)) if (values[i] > 0) != (values[j] > 0)]
        if len(keys) == 2:
            segments.append(keys)

    touching = defaultdict(list)
    for n, keys in enumerate(segments):
        for key in keys:
            touching[key].append(n)

    used = [False] * len(segments)
    chains = []
    for n in range(len(segments)):
        if not used[n]:
            used[n] = True
            chain = list(segments[n])
            for is_forward in (True, False):
                end = chain[-1] if is_forward else chain[0]
                following = [m for m in touching[end] if not used[m]]
                while following:
                    m = following[0]
                    used[m] = True
                    end = segments[m][1] if segments[m][0] == end else segments[m][0]
                    if is_forward:
                        chain.append(end)
                    else:
                        chain.insert(0, end)
                    following = [k for k in touching[end] if not used[k]]

            chains.append(stack([crossings[key] for key in chain]))

    return chains


def _arrowhead(axes, tail: NDArray, tip: NDArray, length: float, color: RGB, zorder: float) -> None:
    """A filled, slightly notched pen arrowhead at tip."""
    direction = (tip - tail) / norm(tip - tail)
    side = asarray([-direction[1], direction[0]])
    base = tip - length * direction
    notch = tip - 0.72 * length * direction
    outline = asarray([tip, base + 0.36 * length * side, notch, base - 0.36 * length * side])
    axes.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.4,
            joinstyle="miter",
            zorder=zorder + 1e-5,
        )
    )


def _distance_to_polygon(points: NDArray, polygon: NDArray) -> NDArray:
    """Distance from each point to the nearest edge of a closed polygon."""
    starts = polygon
    ends = concatenate([polygon[1:], polygon[:1]])
    edge = ends - starts
    relative = points[:, None, :] - starts[None, :, :]
    fraction = clamp((relative * edge).sum(-1) / (edge * edge).sum(-1), 0.0, 1.0)
    nearest = starts[None] + fraction[..., None] * edge[None]
    distance = norm(points[:, None, :] - nearest, axis=-1).min(axis=1)

    return distance
