# Visualization Architecture

A bird's-eye view of the `morphis.visuals` subpackage.

## The Principle: The Scene Draws What It Is Handed

Visualization is kept separate from the mathematics it shows. The user does all the mathematics (rotations, translations, evolutions, simulations, operations) on morphis elements, outside the visuals. The Scene holds references to those elements, and every `capture(t)` re-reads their current state and redraws them. Nothing in `morphis.visuals` transforms, evolves, or simulates an element.

The only mathematics inside visuals is **depiction**: projecting higher-dimensional elements to 3D (see Projection), and turning each kind of element into something drawable:

| Element handed to the Scene | Drawn as |
|-----------------------------|----------|
| `Vector`, grade 1 | arrows from an origin (one per lot entry) |
| `Vector`, grade 2 | an **oriented disk**: area \|B\| in the bivector's plane, rim, and a circulation arrowhead |
| `Vector`, grade 3 | an **oriented ball**: volume \|T\|, with an equator and arrowhead showing handedness |
| `Frame` | its exact vectors as arrows; with `filled=True`, also the parallelogram or parallelepiped they span |
| `Surface`, `VisualModel` | the mesh, with its vertices re-read every capture |

Higher grades raise `ValueError`.

**Why disks, not parallelograms.** A bivector has a plane, a magnitude, and an orientation, but no shape: infinitely many pairs (u, v) give the same u ∧ v. Drawing a bare bivector as a parallelogram would invent factors it does not carry, and any rule that picks them from B alone must jump somewhere, since no direction can be chosen continuously in every plane through the origin (the hairy-ball theorem). The disk carries exactly what B carries and is symmetric in its plane, so a smoothly turning bivector gives a smoothly turning disk. When a specific u ∧ v is meant, hand the Scene those vectors as a `Frame(u, v)` with `filled=True`; it then draws that u, that v, and their parallelogram, following them exactly.

**Stateless depiction.** Each frame is computed from the element alone; the Scene stores no memory of earlier frames. The one choice that needs a direction, where the circulation arrowhead sits on a disk's rim, uses the rim point farthest along world +z. That point depends only on the plane and changes continuously with it, except when the plane faces +z head-on (a horizontal disk), where it falls back to +x. A ball's equator lies in the horizontal plane; its arrowhead runs counterclockwise about +z for T > 0 (right-handed, like e₁₂₃) and clockwise for T < 0. The geometry lives in `drawing/blades.py` (`oriented_disk`, `oriented_ball`) as plain arrays.

## Module Structure

```
src/morphis/visuals/
├── __init__.py       # Public API exports
├── scene.py          # Scene: static, live, and recorded visualization
├── recording.py      # Recording: the shared .mp4 / .gif writer
├── canvas.py         # Canvas: immediate-mode drawing primitives
├── model.py          # VisualModel: mesh whose vertices are GA Vectors
├── text.py           # Text, TextStyle: 3D text annotations (data only)
├── theme.py          # Colors, palettes, themes, window sizes
├── projective.py     # Projection (projective maps), View: Orthographic, Perspective, Stereographic
├── projection.py     # Basis labels; ProjectionConfig and project_blade for the Canvas path
├── contexts.py       # PGA interpretation (points, lines, planes)
├── operations.py     # Meet, join, and dual visualizations
├── backends/         # Rendering backend abstraction used by Scene
│   ├── protocol.py   # RenderBackend protocol
│   └── pyvista.py    # PyVistaBackend implementation
├── drawing/          # Depiction geometry and meshes
│   ├── blades.py     # Oriented disks (bivectors) and balls (trivectors)
│   └── vectors.py    # Arrow, span, frame, tesseract meshes; draw/render helpers
├── ink/              # Pen-and-ink conceptual figures (matplotlib)
│   ├── sketch.py     # Sketch: records marks in 3D, renders them to the page
│   ├── animate.py    # animate(): one sketch per time -> MP4 or GIF
│   ├── space.py      # OrganicSpace: seeded enclosing space and its silhouette
│   ├── camera.py     # Camera: orbiting view, orthographic or perspective
│   ├── theme.py      # InkTheme: paper, ink, graphite, accents, font
│   └── fonts.py      # Font registration, including .ttc italic faces
└── tests/            # test_scene*.py, test_projection.py, test_model.py, test_ink.py
```

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│  High-Level API                                             │
│  Scene (live and recorded), Canvas                          │
│  contexts (PGA), operations (meet/join/dual)                │
├─────────────────────────────────────────────────────────────┤
│  Depiction                                                  │
│  projective: View -> Metric -> Projection to the drawing    │
│  drawing/blades.py: oriented disks and balls                │
├─────────────────────────────────────────────────────────────┤
│  Backend Abstraction                                        │
│  RenderBackend protocol, PyVistaBackend                     │
├─────────────────────────────────────────────────────────────┤
│  PyVista / VTK                                              │
└─────────────────────────────────────────────────────────────┘

            Recording (.mp4 / .gif): shared by Scene and ink
```

Scene talks only to the backend protocol. Canvas, `contexts.py`, and `operations.py` draw directly on a PyVista `Plotter`. The `ink` subpackage is a separate path for conceptual figures rather than shaded 3D scenes: it projects through its own camera and draws pen strokes and stipple with matplotlib.

## Two Ways to Draw

| | Scene | Ink sketch |
|---|---|---|
| For | shaded 3D scenes, live or recorded | pen-and-ink conceptual figures |
| Renderer | PyVista/VTK through `RenderBackend` | matplotlib |
| Motion | mutate elements, then `capture(t)` | `build(t)` returns the figure at time t |
| Projection | `View` or `Projection` | `View` or `Projection` |
| Output | window; `record()` to .gif / .mp4; `save()` to .scene / .obj | `save()` to PNG/SVG/PDF; `animate()` to .gif / .mp4 |
| Size | pixels (`size=(1280, 800)`) | inches (`size=(8, 6)`) at a dpi |
| Frame rate | `frame_rate` | `frame_rate` |

Both keep the mathematics outside the drawing.

## Scene (`scene.py`)

```python
from morphis.visuals import Scene, RED

# Static display
scene = Scene(theme="obsidian")
scene.add(v, color=RED)
scene.show()

# Live animation: the math happens outside, the Scene redraws
scene = Scene(theme="obsidian")
scene.add(F, color=RED, filled=True)
scene.fade_in(F, t=0.0, duration=0.5)
for t in times:
    F.data[...] = F.transform(rotor(b, d_angle)).data
    scene.capture(t)
scene.show()

# Recording, off screen and as fast as frames render
scene = Scene(window=False)
scene.add(B)
with scene.record("figures/turn/turn.gif"):
    for t in times:
        B.data[...] = turned(t)
        scene.capture(t)
```

**Constructor:** `Scene(projection=None, theme="obsidian", size=(1280, 800), frame_rate=30, backend="pyvista", show_basis=True, auto_camera=True, window=True)`. The backend is created at construction and initialized lazily on the first `add()`, `camera()`, `capture()`, or `show()`. With `window=False` it renders off screen: no window opens, `capture` does not wait for wall-clock time, and `show()` does nothing.

**Key methods:**
- `add(element, color=None, opacity=1.0, **options)`: add an element; returns an element ID. Options: `origin` (single element), `origins` (one per lot entry), `filled` (Frame), `smooth_shading` and `show_edges` (meshes)
- `remove(element_id)`
- `capture(t)`: redraw every element from its current state; with a window, keep pace with wall-clock time `t`; inside `record()`, append one frame
- `record(path, frame_rate=None)`: context manager; every capture inside the block becomes a frame of `path` (.mp4 or .gif), finalized on exit, even if the block raises. The frame rate defaults to the scene's. Yields the `Recording`, whose `frame_count` and `written_count` tell what was captured and written
- `set_projection(projection)`: draw through a new `View` or `Projection` (see Projection); every element already in the scene is redrawn through it
- `show()`: show the window and block until it is closed
- `fade_in(element, t, duration)`, `fade_out(element, t, duration)`: schedule opacity effects
- `camera(position, focal_point, up)`, `reset_camera()`, `set_clipping_range(near, far)`
- `add_light(...)`, `remove_light(light_id)`, `clear_lights()`
- `save(path)`, `Scene.load(path, window=True)`: persist and restore
- `close()`, `is_closed()`

**Properties:** `theme`, `frame_rate`, `window`, `projection` (the `View` or `Projection`), `basis_labels`.

**Update path.** Each element's depiction is a fixed set of named shapes (for a bivector: `fill/0`, `rim/0`, `sense/0`), determined by its type, grade, and lot. `add()` creates one backend object per shape. `capture(t)` recomputes the shapes from the element's current data and moves each backend object to its new geometry (`update_arrows`, `update_lines`, `update_mesh`, or `replace_mesh` when the topology can change, as for frames), then applies effect opacity.

## Projection (`projective.py`)

The metric decides the geometry; the projection decides the picture. Elements of any dimension are drawn through a projective map to the 3D drawing space, and both renderers, the Scene and the ink Sketch, take the same two kinds of setting.

**Views.** A `View` names the kind of projection and the three directions that become x, y, z, by their labels, the same labels accepted by `basis_vector` and `.on[...]` (see [Index Convention](6_index-convention.md)). It holds no metric: the Scene is built before it sees any element, so each element resolves the view against its own metric with `view.resolve(metric)`, which returns a `Projection`.

```python
from morphis.visuals import Orthographic, Perspective, Scene, Stereographic

scene = Scene(projection=Orthographic(axes=(1, 2, 3)))            # the default
scene.set_projection(Orthographic(axes=(2, 3, 4)))               # e_2, e_3, e_4
scene.set_projection(Perspective(axes=(1, 2, 3), distance=3.0))  # an eye on the e_4 axis
scene.set_projection(Stereographic(axes=(1, 2, 3), pole=4))      # from the pole of the unit sphere
```

| View | Map | Parameters |
|------|-----|------------|
| `Orthographic(axes)` | the components along the axes; every other direction dropped | |
| `Perspective(axes, distance)` | $\mathbf{x} \mapsto D \, \mathbf{x}_{\text{axes}} / (D - Σ x_{\text{dropped}})$ | `distance` $D$ of the eye |
| `Stereographic(axes, pole, radius)` | from the pole of the sphere of that radius onto the hyperplane through the origin, then orthographic onto the axes | `pole` label (last direction by default), `radius` |

The perspective weight row is $1 - Σ x_{\text{dropped}} / D$, so its center of projection is the flat where the axes components vanish and $Σ x_{\text{dropped}} = D$: from ℝ⁴ the eye sits at $D \, \mathbf{e}_4$, and the far cell of a tesseract is drawn inside the near one. Every view draws a space of dimension 3 or less in its own coordinates, zero-padded. The axes must be three distinct labels of the element's space, so an axis outside it raises `IndexError` (0 is forbidden in a Euclidean space and is time or the ideal direction in a Lorentzian or PGA space), and a stereographic pole may not also be an axis. The default `axes=(1, 2, 3)` is x, y, z in every signature, because x is always 1. Views are frozen pydantic models, compared by value and saved with a scene.

**Projections.** `Projection` holds the $(m + 1) \times (d + 1)$ matrix of a projective map $\mathbb{RP}^d \to \mathbb{RP}^m$ on homogeneous coordinates; its kernel is the center of projection. A projection to 3D may be given wherever a view is, for a map no view names: a screen tilted out of the basis, a deliberately unfaithful linear depiction, or a chain of maps.

| Constructor or method | Role |
|------|------|
| `Projection.linear(metric, images)` | each labeled direction sent to a chosen picture vector; the deliberately unfaithful depiction, in which two directions may share one image |
| `Projection.orthographic(screen)` | picture coordinate $c$ is $\mathbf{e}_c \cdot \mathbf{x}$ for an orthonormal screen `Frame` |
| `Projection.stereographic(pole)` | the sphere of the pole's radius, from the pole |
| `P @ Q` | composition, first Q then P |
| `P(points)` | picture coordinates of points of any lot |
| `P.pushforward(points, vectors)` | the differential $dΠ$ on tangent vectors |
| `P.blade(b, at)` | a k-vector at a point carried by the outermorphism $∧^k dΠ$ |
| `P.pinching(points, frame)` | $s_{\min} / s_{\max}$ of $dΠ$ on a tangent frame, 0 at a pinch point |
| `P.is_affine` | whether the weight is constant, so displacements have images of their own |

`resolve(projection, metric)` is the single entry the renderers use: a view is resolved against the metric, and a given projection is checked to draw that space in 3D.

**What is mapped how.** Points, and everything built from points, are mapped point by point: arrow tails and tips, the $2^k$ corners of a frame's cell, mesh vertices, and ink curves. A projective map sends lines to lines and flats to flats, so the edges and faces drawn between mapped corners are exact under perspective too, where a parallelogram is drawn as a general quadrilateral. A bare blade carries no points beyond where it is attached, so the Scene carries it into the picture by `P.blade` at its origin and draws the result by its invariants (an oriented disk or ball). Under an orthographic view onto basis directions, `P.blade` is the sub-array of components at those directions.

Ink marks that also take displacements (a vector's arrow, a plane's spanning vectors, a circle's plane) set them out from a drawing-space placement, which needs an affine projection; given a central one, `Sketch.carried` raises `ValueError`. Points, curves, surfaces, and labels go through any projection.

**Labels.** `basis_labels(axes)` in `projection.py` gives the LaTeX labels of the drawn basis, $\mathbf{e}_2, \mathbf{e}_3, \mathbf{e}_4$ for axes `(2, 3, 4)`. The Scene passes them to the backend before the basis is first drawn and again on every `set_projection`. A given `Projection` has no axes, and the basis is labeled 1, 2, 3.

**The Canvas path.** `ProjectionConfig` and `project_blade` in `projection.py` remain for the Canvas helpers in `drawing/vectors.py`, `contexts.py`, and `operations.py`, which slice components at chosen axes. They are to move onto views as those helpers are revisited.

## Recording (`recording.py`)

`Recording(path, frame_rate)` is the one writer behind `Scene.record` and `ink.animate`. Frames go in as RGB image arrays; the suffix picks the format, `.mp4` (H.264 via imageio-ffmpeg) or `.gif` (looping). It is a context manager that finalizes the file on exit.

GIF frame delays are whole centiseconds, and browsers slow any delay under 20 ms, so a GIF plays at most 50 frames per second. The writer uses the nearest delay of at least 20 ms and, when frames arrive faster, keeps the frames that fall on the GIF's own ticks, so the clip keeps its real duration. Video writes every frame at the given rate.

The example animations default to `.gif`, which plays inline wherever the docs are read; `.mp4` remains a suffix away. A GIF of stippled ink grows quickly with resolution, so the ink examples render their animation frames at a lower `dpi` than their stills.

## Ink Sketches (`ink/`)

Pen-and-ink conceptual figures in the style of a hand-drawn monograph illustration: an organic enclosing space shaded with stipple, and vectors, planes, curves, construction lines, and labels drawn inside it. Output is a static image (PNG, SVG, PDF).

The design separates **what is true** from **how it is drawn**:

- **True geometry** is ordinary morphis objects in their own space and dimension, with their exact relationships (for example a realified qubit in ℝ⁴ and the exact shadows of a state in two conjugate planes).
- **The projection** carries the true space into a 3D drawing space: a `View` as for the Scene, or a `Projection`. The figure's deliberately unfaithful linear map is `Projection.linear`, stated once by naming the drawing direction that stands for each basis direction. Keys are labels of the true space's `Metric`, so `1` is `e_1`, and two true directions may share one drawing direction when the figure needs it. The default is `Orthographic()`.
- **Placement**: a `Vector` passed to a mark is true geometry and goes through the projection; a plain `(x, y, z)` triple is a drawing-space position chosen by the figure, such as where a plane is set out.
- **`Camera`** orbits a focal point (`azimuth`, `elevation`, `distance`, `fov`; `fov=0` is orthographic) and maps drawing coordinates to the page plus depth.

```python
from morphis.elements import basis_vectors, euclidean_metric
from morphis.visuals import Projection
from morphis.visuals.ink import Camera, OrganicSpace, Sketch

g = euclidean_metric(4)
e_a, f_a, e_b, f_b = basis_vectors(g)  # e_m and its partner e_ṁ (as f_m)
projection = Projection.linear(g, {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})

sketch = Sketch(camera=Camera(azimuth=-24, elevation=34), projection=projection)
sketch.space(OrganicSpace(seed=3, stretch=(1.6, 1.25, 1.05)))
sketch.plane(e_a, f_a, at=(-4.0, -0.8, -0.9), span=((0, 2.6), (0, 2.6)), grid=6)
sketch.vector(e_a + f_b, label="$ψ$")
sketch.save("figures/example.png")
```

**Marks.** `space`, `plane` (stippled parallelogram with a dashed grid), `vector` (pen arrowhead, optional label beside the shaft midpoint), `line` (dashed construction line), `circle` (circle or arc in a plane, optional arrow), `curve` (polyline through points or a lot-shaped `Vector`, solid or dotted), `shell` (a translucent mesh in pen and stipple: its contour inked, lighter with depth, and its body stippled densest where it turns edge-on, with no grid), `surface` (an opaque shaded mesh painted far to near), `point`, and `label`. Marks are recorded in drawing coordinates and rendered only on `save`/`render`, so page bounds, stroke scale, and depth order come from the whole figure. Order is by layer (space, planes, construction lines and curves, vectors, points, labels), then far to near.

**Organic space.** `OrganicSpace` is a star-shaped surface whose radius along a unit direction u is `1 + lumpiness · Σ a_k cos(π f_k (u · d_k) + φ_k) / Σ a_k`, with seeded random directions d_k, frequencies f_k up to `detail`, phases φ_k, and amplitudes a_k ∝ 1/f_k. It is then scaled per axis by `stretch` and by `size`. The silhouette is computed for the camera by projecting a Fibonacci-spiral sampling of the surface and keeping the farthest point at each angle about the projected center, so it stays correct as the camera moves. Shading is rim stipple: dot density falls off exponentially inward from the silhouette and is heavier on a chosen shadow side.

**Repeatability.** Every random choice (shape, stipple, pen wobble) flows from a seed, so the same figure renders byte-for-byte identically and its space does not shift as contents are edited.

**Themes and fonts.** `INK` (white paper), `PARCHMENT` (warm), and `CHALKBOARD` (inverted) share muted accents (blue, red, green, sepia, violet, ochre). Text and mathtext are set in the theme's `font` (default Palatino). System families shipped as `.ttc` collections expose only their upright face to matplotlib, so `fonts.py` extracts each face once to `~/.cache/morphis/fonts` and registers it, making the italic available to math.

**Animation.** `animate(build, times, path, frame_rate=30)` renders one sketch per time, where `build(t)` constructs the figure from the true geometry at time t. This mirrors the Scene principle: the math happens in `build`, outside the drawing. All frames share one page rectangle (the union of every frame's bounds, fitted to the aspect), so still parts stay registered. Each space and mark draws its stipple and pen wobble from its own seeded stream, keyed by the order it was added, so a mark looks the same in every frame even as other marks move or change depth order. Frames go through the shared `Recording` (see Recording), so `.mp4` and `.gif` behave exactly as they do for `Scene.record`. Expensive true-geometry work that does not depend on t, such as a whole orbit, should be computed once and reused across frames.

Not yet integrated with `Scene`: there is no ink `RenderBackend`, and hidden-line dashing is per mark rather than per segment.

## Other Components

### Canvas (`canvas.py`)

Immediate-mode drawing on a PyVista plotter: `arrow`, `arrows`, `curve`, `curves`, `point`, `points`, `plane`, `model`, `camera`, `show`, `screenshot`. Used by `drawing/vectors.py` render helpers, `contexts.py`, and `operations.py`.

```python
canvas = Canvas(theme="obsidian")
canvas.arrow([0, 0, 0], [1, 0, 0])
canvas.show()
```

`Canvas(basis_axes=(1, 2, 3))` and `set_basis_axes(axes)` take user-facing geometric indices and label the axes with the same numbers (`(2, 4, 5)` gives e2, e4, e5). Canvas holds no metric, so it does not range-check these indices.

### VisualModel (`model.py`)

A mesh whose vertices are a lot of grade-1 `Vector`s in 3D, with triangle faces. It is an `Element`, so GA transforms (`apply_similarity`, rotors, motors) act on its vertices directly, outside the Scene. `Scene.add(model)` draws the mesh and every capture re-reads the vertices. Load with `VisualModel.from_file(path)` or `VisualModel.from_mesh(polydata)`.

### Text (`text.py`)

`Text` holds a string, a 3D position, and font settings; `TextStyle` holds reusable styling. The backend protocol has `add_text` / `update_text`, though Scene does not yet route `Text` to them.

### PGA Context (`contexts.py`)

Interprets PGA blades as geometric entities and draws them on a Canvas:

| Grade | Interpretation | Rendering |
|-------|---------------|-----------|
| 1     | Point/direction | Sphere |
| 2     | Line | Extended segment |
| 3     | Plane | Transparent surface |

Entry points: `visualize_pga_blade`, `visualize_pga_scene`, `render_pga_point`, `render_pga_line`, `render_pga_plane`, `is_pga_context`.

### Operation Visualization (`operations.py`)

`render_join`, `render_meet`, `render_meet_join`, and `render_with_dual` draw the inputs and the result of an operation together on a Canvas, styled by `OperationStyle`. Each accepts an optional `ProjectionConfig` for blades of dimension greater than 3.

## Theme System (`theme.py`)

Four built-in themes:

| Theme    | Background | Character |
|----------|------------|-----------|
| obsidian | Charcoal   | Coral, seafoam, amber |
| paper    | Cream      | Rust, teal, ochre |
| midnight | Near-black | Peach, aqua, gold |
| chalk    | Cool gray  | Crimson, teal, marigold |

Each `Theme` provides a background color, basis colors `e1`, `e2`, `e3`, an object `Palette` for cycling, `accent`, `muted`, and `label` colors, plus derived `axis_color`, `grid_color`, and `text_color`. `get_theme(name)` and `list_themes()` look themes up; `Theme.from_background()` derives a theme from a background color. Named colors (`RED`, `BLUE`, ...) are exported alongside.

### Window Size Constants

All sizes use a 16:10 aspect ratio:

```python
from morphis.visuals import SMALL, MEDIUM, LARGE, DEFAULT_SIZE

SMALL = (800, 500)
MEDIUM = (1280, 800)
LARGE = (1920, 1200)

DEFAULT_SIZE = MEDIUM
```

Scene defaults to `MEDIUM`; Canvas defaults to `(1200, 900)`.

## Backend Abstraction (`backends/`)

Scene renders through a pluggable backend obtained with `get_backend(name)`. The `RenderBackend` protocol:

```python
class RenderBackend(Protocol):
    # Lifecycle
    def initialize(size, theme, show_basis=True, window=True): ...
    def close(): ...

    # Objects (add_* returns an object ID)
    def add_mesh(vertices, faces, color, opacity=1.0, smooth_shading=True, show_edges=False): ...
    def update_mesh(object_id, vertices): ...
    def replace_mesh(object_id, vertices, faces): ...
    def add_arrows(origins, directions, color, opacity=1.0, ...): ...
    def update_arrows(object_id, origins, directions): ...
    def add_points(positions, color, opacity=1.0, point_size=5.0): ...
    def update_points(object_id, positions): ...
    def add_lines(points, color, opacity=1.0, line_width=2.0): ...
    def update_lines(object_id, points): ...
    def add_span(origin, vectors, color, opacity=0.3, filled=True): ...
    def update_span(object_id, origin, vectors): ...
    def add_text(text, position, color, font_size=12, anchor="center"): ...
    def update_text(object_id, text=None, position=None): ...
    def set_opacity(object_id, opacity): ...
    def remove(object_id): ...

    # Camera
    def set_camera(position=None, focal_point=None, up=None): ...
    def reset_camera(): ...

    # Rendering and window
    def render(): ...
    def capture_frame() -> NDArray: ...
    def export_obj(path): ...
    def show(interactive=True): ...
    def process_events(): ...
    def is_closed() -> bool: ...
    def wait_for_close(): ...

    # Basis display and lighting
    def set_basis_labels(labels): ...
    def add_light(position, focal_point, intensity, color, directional, attenuation): ...
    def remove_light(light_id): ...
    def clear_lights(): ...
```

Only `PyVistaBackend` is implemented. Labels passed to `set_basis_labels` before `initialize` are used when the basis is first drawn. Scene uses nothing outside the protocol.

## Saving and Loading Scenes

```python
scene.save("demo.scene")  # Pickle format, reloadable
scene.save("demo.obj")    # Wavefront OBJ, opens in macOS Preview

scene = Scene.load("demo.scene")              # on screen
scene = Scene.load("demo.scene", window=False)  # off screen, e.g. to record or export
scene.show()
```

| Extension | Format | Use case |
|-----------|--------|----------|
| `.scene` | Pickle of `SceneData` | Reloadable with theme, size, projection (the `View` or `Projection`), basis flag, and per-element color, opacity, and options |
| `.obj` | Wavefront OBJ | View in 3D apps (macOS Preview, Blender, etc.) |

**CLI viewing:**

```bash
morphis view demo.scene
```

## Rendering Flow

### Live and Recorded (Scene)

```
User code                         Scene
    │                               │
    │  scene.add(element)           │
    ├──────────────────────────────►│ Depicts the element, creates backend objects
    │                               │
    │  element.data[...] = new      │   (the math, outside the Scene)
    │  scene.capture(t)             │
    ├──────────────────────────────►│ Re-depicts every element from its current state
    │                               │ Moves backend objects, applies effect opacity
    │                               │ Inside record(): appends a frame
    │                               │ With a window: waits for wall-clock t
    │                               │
    │  scene.show()                 │
    ├──────────────────────────────►│ Waits for window close (window only)
```

### Static Display

```
scene.add(element)  →  Creates visuals
scene.show()        →  Blocking show, waits for close
```

## Design Decisions

1. **The Scene draws what it is handed:** no transformation, simulation, or evolution inside visuals; depiction only.

2. **Stateless depiction:** each frame is computed from the element alone, so pictures follow elements continuously and nothing is remembered between frames.

3. **Bare blades are drawn by their invariants:** an oriented disk for a bivector and an oriented ball for a trivector; a specific factorization is drawn by handing over a `Frame`.

4. **Recording wraps the capture loop:** `with scene.record(path):` around ordinary captures, with one writer shared by Scene and ink.

5. **Backend abstraction:** Scene depends on the `RenderBackend` protocol only.

6. **Metric-routed projection:** a view names its axes by label and resolves against each element's `Metric`; basis labels use the same labels. No visuals code converts indices with an ad-hoc offset.

7. **Real-time sync on screen, full speed off screen:** with a window, `capture(t)` waits for wall-clock time `t`; with `window=False` it does not.

8. **Truth separate from depiction (ink):** figures keep objects in their true space and state every unfaithful drawing choice in one projection plus explicit placements.

## Extension Points

| To add... | Modify... |
|-----------|-----------|
| New theme | `THEMES` dict in `theme.py` |
| New backend | Implement `RenderBackend`, register in `backends/__init__.py` |
| New element type | `Scene._depict()`: return named `Shape`s for it |
| New depiction geometry | `drawing/blades.py` (plain arrays, no renderer) |
| New output format | `recording.py` |
| New projection | A `View` subclass in `projective.py` whose `_project` builds the matrix from labels through the metric |
| New ink mark | A recorder method on `Sketch`, a `LAYERS` entry, and a `_render_<kind>` method |
| New ink theme | An `InkTheme` in `ink/theme.py`, added to `INK_THEMES` |
