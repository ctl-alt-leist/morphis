"""
Sketch

A pen-and-ink figure. Marks are recorded in 3D drawing coordinates as they are
added and rendered to the page only on save, once the whole figure is known,
so the layout, the stroke scale, and the depth ordering are all settled from
the complete scene.

Geometry passed as a morphis Vector is true-space geometry, carried into the
drawing through the sketch's Depiction. Geometry passed as a plain triple is a
drawing-space placement: where the figure chooses to put something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib import rc_context
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import withStroke
from numpy import (
    arange,
    arctan2,
    asarray,
    clip as clamp,
    concatenate,
    cos,
    cumsum,
    exp as exponential,
    interp,
    linspace,
    meshgrid,
    pi,
    sin,
    sqrt,
    stack,
)
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.typing import NDArray

from morphis.elements.vector import Vector
from morphis.visuals.ink.camera import Camera
from morphis.visuals.ink.depiction import Depiction
from morphis.visuals.ink.fonts import font_settings
from morphis.visuals.ink.space import OrganicSpace
from morphis.visuals.ink.theme import RGB, InkTheme, get_ink_theme


Point = Vector | tuple[float, float, float] | NDArray

# Draw order of mark kinds; within a layer, farther marks are drawn first
LAYERS = {"space": 1, "plane": 3, "construction": 5, "curve": 5, "vector": 6, "point": 7, "label": 9}

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
        depiction: Map from the true space into the drawing space
        size: Figure size in inches
        seed: Seed for stipple and pen wobble, so renders are repeatable
    """

    def __init__(
        self,
        camera: Camera | None = None,
        theme: str | InkTheme = "ink",
        depiction: Depiction | None = None,
        size: tuple[float, float] = (8.0, 6.0),
        seed: int = 0,
    ):
        self.camera = camera or Camera()
        self.theme = get_ink_theme(theme)
        self.depiction = depiction
        self.size = size
        self.seed = seed
        self.marks: list[Mark] = []
        self.spaces: list[tuple[OrganicSpace, dict]] = []

    # =========================================================================
    # Coordinates
    # =========================================================================

    def drawn(self, point: Point) -> NDArray:
        """Drawing coordinates of a true-space Vector or a drawing-space triple."""
        coordinates = self.depiction(point) if isinstance(point, Vector) else asarray(point, dtype=float)

        return coordinates

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
        origin, du, dv = self.drawn(at), self.drawn(u), self.drawn(v)
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
        tip = tail + scale * self.drawn(v)
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
        du, dv = self.drawn(u), self.drawn(v)
        du, dv = du / norm(du), dv / norm(dv)
        angles = linspace(arc[0], arc[1], 160)
        points = self.drawn(center) + radius * (cos(angles)[:, None] * du + sin(angles)[:, None] * dv)
        style = {"weight": weight, "level": level, "arrow": arrow}
        self.marks.append(Mark("curve", points, style))

    def point(self, p: Point, radius: float = 3.2, color: str | RGB | None = None) -> NDArray:
        """Draw a solid dot; returns its drawing coordinates."""
        position = self.drawn(p)
        self.marks.append(Mark("point", position[None, :], {"radius": radius, "color": color}))

        return position

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

    def render(self, dpi: int = 200) -> Figure:
        """Render the sketch to a matplotlib Figure."""
        figure = Figure(figsize=self.size, dpi=dpi, facecolor=self.theme.paper)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        axes.set_facecolor(self.theme.paper)

        outlines = [(space.outline(self.camera), options) for space, options in self.spaces]
        page_points = [outline for (outline, _), _ in outlines] + [self.camera.project(m.points)[0] for m in self.marks]
        everything = concatenate(page_points) if page_points else asarray([[0.0, 0.0]])
        lower, upper = everything.min(axis=0), everything.max(axis=0)
        margin = 0.06 * (upper - lower).max()
        lower, upper = lower - margin, upper + margin

        # Fit the page bounds to the figure aspect so one unit is equal in x and y
        width, height = self.size
        extent = upper - lower
        scale = max(extent[0] / width, extent[1] / height)
        middle = 0.5 * (lower + upper)
        half = 0.5 * scale * asarray([width, height])
        axes.set(xlim=(middle[0] - half[0], middle[0] + half[0]), ylim=(middle[1] - half[1], middle[1] + half[1]))
        axes.set_aspect("equal")

        units_per_point = scale / 72.0
        rng = default_rng(self.seed)

        for (outline, center), options in outlines:
            self._render_space(axes, outline, center, options, units_per_point, rng)

        depths = [self.camera.project(m.points)[1].mean() for m in self.marks]
        ordered = sorted(zip(self.marks, depths, strict=True), key=lambda pair: (LAYERS[pair[0].kind], -pair[1]))
        for order, (mark, _) in enumerate(ordered):
            zorder = LAYERS[mark.kind] + order * 1e-4
            page, _ = self.camera.project(mark.points)
            renderer = getattr(self, f"_render_{mark.kind}")
            renderer(axes, page, mark, zorder, units_per_point, rng)

        return figure

    def save(self, path: str | Path, dpi: int = 200) -> Path:
        """Render and write the sketch; the format follows the extension (png, svg, pdf)."""
        target = Path(path).expanduser()
        fonts = font_settings(self.theme.font)
        with rc_context(fonts):
            figure = self.render(dpi=dpi)
            figure.savefig(target, dpi=dpi, facecolor=self.theme.paper)

        return target

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

        edge = _wobble(concatenate([page, page[:1]]), 0.5 * units_per_point, rng, samples=200)
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
        axes.plot(*page.T, color=ink, linewidth=WEIGHTS[style["weight"]], solid_capstyle="round", zorder=zorder)
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
