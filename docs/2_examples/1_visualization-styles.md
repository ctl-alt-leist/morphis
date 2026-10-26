# Visualization Styles

Every example in this section draws a space of more than three dimensions on a page or a screen. The mathematics is done first, on ordinary morphis elements in their own space, and a drawing is a separate step that only depicts what it is handed. morphis has two renderers for that step, and both carry a space of any dimension into the three-dimensional drawing space through the same projections. This page describes the two renderers, the projections they share, and how each kind of element passes through a projection. The examples that follow each state which projection they use and what it preserves.

## Two Renderers

The `Scene` draws shaded 3D geometry with PyVista, either live in a window or recorded off screen. It holds references to the user's elements, and each `capture(t)` re-reads their current state and redraws them. The rotations, evolutions, and simulations all happen outside the Scene; the only mathematics inside it is depiction. A turning frame is animated by changing the frame and capturing again, as in the tesseract example:

```python
from morphis.visuals import RED, Orthographic, Perspective, Scene

scene = Scene(frame_rate=30, window=output is None, projection=Orthographic(axes=(1, 2, 3)))
scene.add(edges, color=RED, filled=True, origin=corner)

def turn(angle: float) -> None:
    R = rotor(B, angle)                                   # the mathematics, outside the Scene
    edges.data[...] = start_edges.transform(R).data
    corner.data[...] = (R * start_corner * ~R)[1].data

with scene.record(output):
    for n in range(n_turn + 1):
        turn(start + total * ease_in_out_cubic(n / n_turn))
        scene.capture(t)
        t += dt
```

Each element type has a fixed depiction. A grade-1 `Vector` is drawn as arrows from an origin, a `Frame` as its exact vectors with the parallelogram or parallelepiped they span when `filled=True`, and a mesh as its vertices re-read on every capture. A bare bivector $b$ has a plane, a magnitude, and an orientation, and no preferred pair of factors, so it is drawn as an oriented disk of area $|b|$ in its plane with a circulation arrow. A trivector $w$ is drawn as an oriented ball of volume $|w|$ with its handedness marked on the equator. A depiction is computed from the element alone at each frame, so a smoothly changing element gives a smoothly changing picture.

The ink `Sketch` draws pen-and-ink conceptual figures with matplotlib: stippled enclosing spaces, planes with dashed grids, arrows, construction lines, curves, and labels. Marks are recorded in 3D drawing coordinates as they are added and rendered to the page on `save`, once the whole figure is known, so the page bounds, the stroke scale, and the depth order come from the complete figure. A still figure is a function that builds a sketch; an animation is that function evaluated at each time, and `animate` writes the frames:

```python
from morphis.visuals.ink import Camera, Sketch, animate

def create_sketch(theme: str = "ink", t: float = 0.0) -> Sketch:
    state = evolve(INITIAL_STATE, t)
    sketch = Sketch(camera=Camera(azimuth=-24.0, elevation=34.0, distance=16.0, fov=16.0), projection=projection)
    ...
    return sketch

animate(lambda t: create_sketch(theme, t), times, output, frame_rate=FRAME_RATE, dpi=ANIMATION_DPI)
```

All frames of an ink animation share one page rectangle, the union of every frame's bounds, so parts held still stay registered. Every stipple dot and pen wobble comes from a seeded stream, so a mark looks the same in every frame even as other marks move.

| | `Scene` | `Sketch` |
|---|---|---|
| For | shaded 3D scenes, live or recorded | pen-and-ink conceptual figures |
| Renderer | PyVista | matplotlib |
| Motion | change the elements, then `capture(t)` | `build(t)` returns the figure at time $t$ |
| Projection | a `View` or a `Projection` | a `View` or a `Projection` |
| Still output | a window, or `save()` to `.scene` or `.obj` | `save()` to PNG, SVG, or PDF |
| Animation | `record(path)` around the capture loop | `animate(build, times, path)` |

## Views

A `View` names the kind of projection and the three directions that become the drawing's $x$, $y$, and $z$, by label, the same labels that `basis_vector` and `.on[...]` accept. It holds no metric: each element resolves the view against its own metric into a `Projection`. A space of dimension three or less is drawn in its own coordinates by every view. There are three views.

`Orthographic(axes)` keeps the components along the three axes and drops every other direction:

```math
\mathbf{x} \mapsto (x^{a}, x^{b}, x^{c})
```

for `axes=(a, b, c)`. It is a linear map, so parallel lines stay parallel, and it is faithful on anything that lies in the span of the axes. Two points that differ only along a dropped direction are drawn at the same place.

`Perspective(axes, distance)` puts an eye at distance $D$ along the dropped directions and draws a point by central projection:

```math
\mathbf{x} \mapsto \frac{D \mathbf{x}_{\text{axes}}}{D - Σ x_{\text{dropped}}}
```

From $\mathbb{R}^4$ the eye sits at $D \mathbf e_4$, so a point with larger $x^4$ is nearer the eye and drawn farther from the center. The dropped coordinate becomes visible as scale, which separates the near and far cells of a tesseract.

`Stereographic(axes, pole, radius)` projects the sphere of radius $ρ$ from its pole $ρ \mathbf{n}$ onto the hyperplane through the origin perpendicular to $\mathbf{n}$, and then orthographically onto the axes:

```math
\mathbf{y} = \frac{ρ \mathbf{x}_\perp}{ρ - \mathbf{n} \cdot \mathbf{x}}
```

On the sphere it is a conformal bijection from everything except the pole onto the whole hyperplane. A circle on the sphere is drawn as a circle, and a circle through the pole as a straight line. It is the natural view of the unit sphere $S^3$ of a qubit's state space, and the pole may not also be one of the axes.

```python
from morphis.elements import Dot
from morphis.visuals.projective import Orthographic, Perspective, Stereographic

Orthographic(axes=(2, 3, 4))                         # e_2, e_3, e_4 as x, y, z
Perspective(axes=(1, 2, 3), distance=3.0)            # an eye at 3 e_4
Stereographic(axes=(1, Dot(1), 2), pole=Dot(2))      # from the pole e_2̇ of the unit sphere
```

A Scene changes its view with `scene.set_projection(view)`, and every element already in it is redrawn through the new one.

## Projections

A `Projection` is a projective map from a space of dimension $d$ to a picture of dimension $m$, an $(m + 1) \times (d + 1)$ matrix $M$ acting on homogeneous coordinates:

```math
\begin{aligned}
(\mathbf{y}, w) &= M (\mathbf{x}, 1) \\
Π(\mathbf{x}) &= \mathbf{y} / w
\end{aligned}
```

The last row of $M$ gives the weight $w$. An orthographic map has a constant weight and an ideal center of projection; a perspective or stereographic map has a weight that varies with $\mathbf{x}$ and a finite center. Every view resolves to one of these, and a `Projection` may be passed wherever a view is, for a map that no view names. Three constructors build them:

| Constructor | Map |
|---|---|
| `Projection.linear(metric, images)` | each labeled basis direction sent to a chosen picture vector |
| `Projection.orthographic(screen)` | picture coordinate $c$ is $\mathbf e_c \cdot \mathbf{x}$ for an orthonormal screen |
| `Projection.stereographic(pole)` | the sphere of radius $|\text{pole}|$, projected from the pole |

`Projection.linear` is the deliberately unfaithful map. Directions left out of `images` are drawn as zero, and two directions may share one image. The conjugate-planes figure uses it to draw both mode directions of a qubit along the same page direction, so that each mode's plane can be laid out on its own:

```python
from morphis.visuals.projective import Projection

projection = Projection.linear(METRIC, {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})
```

Maps compose with `@`, first the right one and then the left, as long as the picture of the first is the space of the second. The qutrit figure reaches the drawing space from $\mathbb{R}^6$ in two steps, an orthographic map onto four directions followed by stereographic projection of a 3-sphere in them:

```python
EXCITED = Projection.orthographic([basis_vector(label, SPACE) for label in (2, Dot(2), 3, Dot(3))])
PROJECTION = Projection.stereographic(RADIUS * basis_vector(4, EXCITED.target)) @ EXCITED
```

`EXCITED.target` is the Euclidean $\mathbb{R}^4$ of the intermediate picture, so the pole is named in that space. The metric may help build a projection, as for an orthonormal screen, but it is never consulted when the projection is applied. `resolve(view, metric)` is the single entry the renderers use: it resolves a view against the metric, or checks that a given projection draws that space in 3D.

## Points and Blades

Points, and everything built from points, are mapped point by point: arrow tails and tips, the $2^k$ corners of a frame's cell, mesh vertices, and ink curves. A projective map sends lines to lines and flats to flats, so the edges and faces drawn between mapped corners are exact under perspective too. A parallelogram is drawn as a general quadrilateral, and a tesseract's cells keep straight edges at every instant of a turn.

A bare blade has no points beyond the one it is attached at, so the Scene carries it into the picture by the differential of the projection there. For $Π(\mathbf{x}) = \mathbf{y} / w$, with linear block $A$ and weight row $\mathbf{c}$, the differential at $\mathbf{x}$ is:

```math
dΠ(\mathbf{v}) = \frac{A \mathbf{v} - Π(\mathbf{x}) (\mathbf{c} \cdot \mathbf{v})}{w}
```

A $k$-vector is pushed forward by the outermorphism $∧^k dΠ$. For a bivector:

```math
b'^{pq} = dΠ^p{}_m dΠ^q{}_n b^{mn}
```

so its plane becomes the image plane and its magnitude the drawn area. The Scene draws the result by its invariants, a disk for a bivector and a ball for a trivector, which carry exactly what the blade carries. Under an orthographic view onto basis directions, the pushforward is the sub-array of components at those directions. `Projection.pushforward` and `Projection.blade` are public, and `Projection.pinching` measures from $dΠ$ how nearly the map collapses a tangent plane, as the [Calabi–Yau slice](6_calabi-yau-slice.md) uses to find its pinch points.

Ink marks that take displacements, such as a vector's arrow, a plane's spanning vectors, or a circle's plane, set them out from a drawing-space placement. They need an affine projection, one whose displacements have images of their own, and `Sketch.carried` raises `ValueError` for a central one. Points, curves, surfaces, and labels pass through any projection. A `Vector` given to a mark is true geometry and goes through the projection; a plain triple is a placement in the drawing space, chosen by the figure:

```python
corner = anchor - reach * (sketch.drawn(e) + sketch.drawn(f))     # a placement built from drawn directions
sketch.plane(e, f, at=corner, span=((0.0, 2 * reach), (0.0, 2 * reach)), grid=6, tone=0.3)
```

## Output

Both renderers write animations through one `Recording`, and the file suffix picks the format: `.gif` is a looping animated GIF and `.mp4` is H.264 video. The examples default to `.gif`, which plays inline wherever these pages are read, and an `.mp4` is one suffix away:

```
uv run python -m morphis.examples.conjugate_planes --animate figures/conjugate-planes/conjugate-planes.mp4
```

GIF frame delays are whole centiseconds and browsers slow any delay under 20 ms, so a GIF plays at most 50 frames per second; when frames arrive faster, the writer keeps the frames that fall on its own ticks, and the clip keeps its duration. The writer chooses one palette for the whole clip and stores each frame as the rectangle that changed, so a still background costs almost nothing after the first frame. A GIF of stippled ink still grows quickly with resolution, so the ink examples render animation frames at a lower `dpi` than their stills.

## The Examples

| Example | Space | Renderer | Projection |
|---|---|---|---|
| [Conjugate planes](2_conjugate-planes.md) | a qubit's two modes, $\mathbb{R}^4$ | `Sketch` | `Projection.linear`, unfaithful by design |
| [Qutrit evolution](3_qutrit-evolution.md) | a spin-1, $\mathbb{C}^3 = \mathbb{R}^6$ | `Sketch` | orthographic to $\mathbb{R}^4$, then stereographic from a 3-sphere; and an equiangular linear map of all six directions |
| [Clifford torus](4_clifford-torus.md) | a qubit, $S^3 \subset \mathbb{R}^4$ | `Sketch` | `Stereographic` |
| [Tesseract](5_tesseract.md) | $\mathbb{R}^4$ | `Scene` | `Orthographic` and `Perspective` |
| [Calabi–Yau slice](6_calabi-yau-slice.md) | a complex curve in $\mathbb{C}^2 = \mathbb{R}^4$ | `Sketch` | `Projection.orthographic` onto a tilted screen |
