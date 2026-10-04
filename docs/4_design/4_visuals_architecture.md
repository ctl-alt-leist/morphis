# Visualization Architecture

A bird's-eye view of the `morphis.visuals` subpackage.

## Module Structure

```
src/morphis/visuals/
├── __init__.py       # Public API exports
├── scene.py          # Scene: static and live-animated visualization
├── loop.py           # Animation: observer/recorder with playback and GIF/MP4 export
├── renderer.py       # Renderer: PyVista actor management used by Animation
├── canvas.py         # Canvas: immediate-mode drawing primitives
├── model.py          # VisualModel: mesh whose vertices are GA Vectors
├── text.py           # Text, TextStyle: 3D text annotations (data only)
├── theme.py          # Colors, palettes, themes, window sizes
├── effects.py        # Scheduled effects (FadeIn, FadeOut, Hold)
├── projection.py     # nD -> 3D projection, index translation, basis labels
├── contexts.py       # PGA interpretation (points, lines, planes)
├── operations.py     # Meet, join, and dual visualizations
├── backends/         # Rendering backend abstraction used by Scene
│   ├── protocol.py   # RenderBackend protocol
│   └── pyvista.py    # PyVistaBackend implementation
├── drawing/          # Mesh generation and blade rendering
│   └── vectors.py    # Arrow, span, frame, tesseract meshes; draw/render helpers
└── tests/            # test_scene.py, test_projection.py, test_model.py
```

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│  High-Level API                                             │
│  Scene, Animation, Canvas                                   │
│  contexts (PGA), operations (meet/join/dual)                │
├─────────────────────────────────────────────────────────────┤
│  Projection                                                 │
│  user geometric indices -> Metric -> internal storage slots │
├──────────────────────────────┬──────────────────────────────┤
│  Backend Abstraction (Scene) │  Renderer (Animation)        │
│  RenderBackend, PyVista      │  actor tracking              │
├──────────────────────────────┴──────────────────────────────┤
│  Drawing Primitives (drawing/vectors.py)                    │
│  create_blade_mesh(), create_frame_mesh(), arrow and spans  │
├─────────────────────────────────────────────────────────────┤
│  PyVista / VTK                                              │
│  Plotter, meshes, actors                                    │
└─────────────────────────────────────────────────────────────┘
```

Scene talks to the backend protocol. Animation predates the backend layer and drives PyVista through `Renderer`. Canvas, `contexts.py`, and `operations.py` draw directly on a PyVista `Plotter`.

## Core Components

### Scene (`scene.py`)

The main interface for both static and animated visualization.

```python
from morphis.visuals import Scene, RED

# Static display
scene = Scene(theme="obsidian")
scene.add(v, color=RED)
scene.show()

# Live animation
scene = Scene(theme="obsidian")
scene.add(F, color=RED, filled=True)
scene.fade_in(F, t=0.0, duration=0.5)

for t in times:
    F.data[...] = transform(t)
    scene.capture(t)  # Renders live, syncs to real time

scene.show()  # Wait for window close
```

**Constructor:** `Scene(projection=None, theme="obsidian", size=(1280, 800), frame_rate=30, backend="pyvista", show_basis=True, auto_camera=True)`. The backend is created at construction and initialized lazily on the first `add()`, `camera()`, `capture()`, or `show()`.

**Key methods:**
- `add(element, representation=None, color=None, opacity=1.0, **kwargs)`: add an element, returns an element ID
- `remove(element_id)`: remove an element
- `set_projection(axes)`: choose the three geometric indices shown (see Projection below)
- `capture(t)`: render the current state at time `t` and wait for wall-clock time `t`
- `show()`: show the window and block until it is closed
- `fade_in(element, t, duration)`, `fade_out(element, t, duration)`: schedule opacity effects
- `camera(position, focal_point, up)`, `reset_camera()`, `set_clipping_range(near, far)`
- `add_light(...)`, `remove_light(light_id)`, `clear_lights()`
- `save(path)`, `Scene.load(path)`: persist and restore (see below)
- `close()`, `is_closed()`

**Properties:** `theme`, `frame_rate`, `projection` (user-facing indices), `basis_labels`.

**Supported elements:**

| Element | Default representation | Backend call |
|---------|------------------------|--------------|
| `Surface` | mesh | `add_mesh` |
| `Frame` | arrows (plus faces when `filled=True`) | `add_mesh` via `create_frame_mesh` |
| `Vector`, grade 1 | arrow (arrows for a lot) | `add_arrows` |
| `Vector`, grade ≥ 2 | span of its factored vectors | `add_span` |

`Text` and `VisualModel` are not drawn by Scene yet. `VisualModel` is supported by `Animation` and `Canvas.model()`.

**Update path:** `capture(t)` calls `_sync_visuals(t)`, which recomputes effect opacity for every element and re-reads the geometry of `Surface`, `Frame`, and grade-1 `Vector` elements. Grade ≥ 2 vectors are factored once in `add()`; their geometry does not refresh on `capture()`, only their opacity does.

**Design principle:** Scene keeps references to the user's elements rather than copies. Animation happens live during `capture()` calls; nothing is recorded.

### Projection (`projection.py`)

Elements of dimension greater than 3 are projected onto three axes for display. Projection axes are **user-facing geometric indices**, the same indices accepted by `basis_vector` and `.on[...]` (see [Index Convention](6_index-convention.md)):

```python
scene = Scene(projection=(1, 2, 3))  # e_1, e_2, e_3 (the default)
scene.set_projection((2, 3, 4))      # switch to e_2, e_3, e_4
```

Translation rules:

- The default `DEFAULT_PROJECTION = (1, 2, 3)` is x, y, z in every signature, because x is always index 1.
- Scene is built before it sees any metric, so it stores the user indices and translates them per element at projection time through `Metric.to_internal_multi`. A 4D Euclidean element maps `(1, 2, 3)` to slots `(0, 1, 2)`; a 3D PGA element (dim 4) maps it to slots `(1, 2, 3)`, skipping the ideal direction.
- An out-of-range index raises `IndexError`. Index 0 is forbidden for a Euclidean element and selects time or the ideal direction for Lorentzian and PGA elements.
- `set_projection` checks the new axes against every element of dimension greater than 3 already in the scene before changing anything.
- Elements of dimension 3 or less are drawn in their own coordinates (zero-padded to 3D) and are not projected.
- Basis labels come from the same indices: `basis_labels((2, 3, 4))` gives e₂, e₃, e₄. Scene passes them to the backend before the basis is first drawn and again on every `set_projection`.

Module contents:

| Name | Role |
|------|------|
| `DEFAULT_PROJECTION` | `(1, 2, 3)` |
| `validate_projection_axes(axes)` | requires exactly three integers, raises `ValueError` otherwise |
| `projection_slots(axes, metric)` | user indices to internal slots via the metric |
| `basis_labels(axes)` | LaTeX labels `$\mathbf{e}_{n}$` for the given indices |
| `ProjectionConfig(axes, method, target_dim)` | blade projection settings; `axes` are user indices, `method` is `"slice"` or `"principal"` |
| `project_blade(blade, config)` | project a grade 0 to 4 blade to `target_dim` |
| `get_projection_axes(blade, config)` | the user indices a projection would show |

The `"principal"` method (also used when `axes` is `None`) picks the internal slots with the largest components, and `get_projection_axes` reports them back through `Metric.to_user`.

Below the projection layer, `Renderer`, `create_blade_mesh`, `create_frame_mesh`, and `create_quadvector_mesh` take a `projection_axes` argument of **internal storage slots**. Their callers (Scene, Animation) translate through the metric first; the drawing code does only array math.

### Animation (`loop.py`)

An observer and recorder. It reads the state of watched objects at each `capture(t)` and either renders immediately (live mode) or stores snapshots for later playback and export.

```python
anim = Animation(frame_rate=60, theme="obsidian")
anim.watch(F, color=RED)
anim.start()             # batch mode; start(live=True) renders as it goes
for t in times:
    F.data[...] = transform(t)
    anim.capture(t)
anim.play()              # or anim.save("rotation.gif") / "rotation.mp4"
```

**Key methods:** `watch(*targets)`, `unwatch(*targets)`, `set_vectors(blade, vectors, origin)`, `set_projection(axes, labels=None)`, `fade_in`, `fade_out`, `start(live)`, `capture(t)`, `finish()`, `play(loop)`, `save(filename, loop)`, `camera(position, focal_point)`, `set_basis_labels(labels)`, `close()`.

Animation accepts `Vector`, `Frame`, and `VisualModel` targets. Blades are factored into spanning vectors on every capture (via `morphis.utils.observer.Observer`), so animated bivectors and trivectors do refresh here. `set_projection` follows the same index convention as Scene: user indices are translated per object through its metric at capture time, and the generated labels use the same indices.

### Renderer (`renderer.py`)

Low-level actor management for Animation. Tracks one set of actors per object ID and rebuilds meshes through `create_blade_mesh` / `create_frame_mesh` on `update_object`. It has no knowledge of geometric algebra, time, or effects.

### Canvas (`canvas.py`)

Immediate-mode drawing on a PyVista plotter: `arrow`, `arrows`, `curve`, `curves`, `point`, `points`, `plane`, `model`, `camera`, `show`, `screenshot`. Used by `drawing/vectors.py` render helpers, `contexts.py`, and `operations.py`.

```python
canvas = Canvas(theme="obsidian")
canvas.arrow([0, 0, 0], [1, 0, 0])
canvas.show()
```

`Canvas(basis_axes=(1, 2, 3))` and `set_basis_axes(axes)` take user-facing geometric indices and label the axes with the same numbers (`(2, 4, 5)` gives e2, e4, e5). Canvas holds no metric, so it does not range-check these indices.

### VisualModel (`model.py`)

A mesh whose vertices are a lot of grade-1 `Vector`s in 3D, with triangle faces. It is an `Element`, so GA transforms (`apply_similarity`, rotors, motors) act on its vertices directly. A cached PyVista mesh shares the vertex buffer; `sync_mesh()` refreshes it at capture boundaries. Load with `VisualModel.from_file(path)` or `VisualModel.from_mesh(polydata)`.

### Text (`text.py`)

`Text` holds a string, a 3D position, and font settings; `TextStyle` holds reusable styling. The backend protocol has `add_text` / `update_text`, though Scene does not yet route `Text` to them.

### Effects (`effects.py`)

Declarative opacity schedules: `FadeIn`, `FadeOut`, `Hold`, and `compute_opacity(effects, object_id, t)`. Animation uses these classes. Scene keeps its own lightweight `SceneEffect`, keyed by element ID, with the same fade semantics.

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

### Theme System (`theme.py`)

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

Scene defaults to `MEDIUM`. Canvas defaults to `(1200, 900)` and Animation to `(1800, 1350)`.

### Backend Abstraction (`backends/`)

Scene renders through a pluggable backend obtained with `get_backend(name)`. The `RenderBackend` protocol:

```python
class RenderBackend(Protocol):
    # Lifecycle
    def initialize(size, theme, show_basis=True): ...
    def close(): ...

    # Objects (add_* returns an object ID)
    def add_mesh(vertices, faces, color, opacity=1.0, smooth_shading=True, show_edges=False): ...
    def update_mesh(object_id, vertices): ...
    def add_arrows(origins, directions, color, opacity=1.0, tip_length=0.1, tip_radius=0.03, shaft_radius=0.015): ...
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

Only `PyVistaBackend` is implemented. Labels passed to `set_basis_labels` before `initialize` are used when the basis is first drawn. Two Scene paths still reach past the protocol into PyVista: Frame updates use `PyVistaBackend.get_actor` and the VTK mapper, and `save("*.obj")` uses the backend's plotter directly.

### Saving and Loading Scenes

```python
scene.save("demo.scene")  # Pickle format, reloadable
scene.save("demo.obj")    # Wavefront OBJ, opens in macOS Preview

scene = Scene.load("demo.scene")
scene.show()
```

| Extension | Format | Use case |
|-----------|--------|----------|
| `.scene` | Pickle of `SceneData` | Reloadable with theme, size, projection (user indices), basis flag, and per-element color, opacity, representation, options |
| `.obj` | Wavefront OBJ | View in 3D apps (macOS Preview, Blender, etc.) |

**CLI viewing:**

```bash
morphis view demo.scene
```

## Rendering Flow

### Live Animation (Scene)

```
User code                         Scene
    │                               │
    │  scene.add(element)           │
    ├──────────────────────────────►│ Projects through element.metric
    │                               │ Creates backend visuals
    │                               │
    │  element.data[...] = new      │
    │  scene.capture(t)             │
    ├──────────────────────────────►│ Updates opacity from effects
    │                               │ Re-projects surfaces, frames, grade-1 vectors
    │                               │ Waits for real time, processes events
    │                               │
    │  scene.show()                 │
    ├──────────────────────────────►│ Waits for window close
```

### Static Display

```
scene.add(element)  →  Creates visuals
scene.show()        →  Blocking show, waits for close
```

### Recorded Animation (Animation)

```
anim.watch(target)     →  Track by id(target)
anim.start()           →  Begin session (batch or live)
anim.capture(t)        →  Snapshot (origin, vectors, opacity, slots) per object
anim.play() / save()   →  Replay snapshots through Renderer or an off-screen plotter
```

## Design Decisions

1. **Scene as the main interface:** one class for static and live-animated scenes.

2. **No snapshot storage in Scene:** animation happens live; Animation is the tool for recording and export.

3. **Backend abstraction:** Scene depends on the `RenderBackend` protocol, not on PyVista directly.

4. **Metric-routed projection:** projection axes and basis labels use user-facing geometric indices, translated per element through its `Metric`. No visuals code converts indices with an ad-hoc offset.

5. **Standard window sizes:** consistent sizing across examples.

6. **Real-time sync:** `capture(t)` waits for wall-clock time `t`.

7. **Clean window close:** windows close without Ctrl-C.

## Extension Points

| To add... | Modify... |
|-----------|-----------|
| New theme | `THEMES` dict in `theme.py` |
| New backend | Implement `RenderBackend`, register in `backends/__init__.py` |
| New element type | `Scene._default_representation()`, `_create_visuals()`, `_sync_visuals()` |
| New projection method | `projection.py` (keep user indices at the API, slots below it) |
