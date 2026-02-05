# Visualization Architecture

A bird's-eye view of the `morphis.visuals` subpackage.

## Module Structure

```
src/morphis/visuals/
├── __init__.py       # Public API exports, window size constants
├── scene.py          # Scene - unified visualization interface (NEW)
├── theme.py          # Color themes and palettes
├── canvas.py         # Low-level drawing canvas
├── renderer.py       # Blade rendering
├── loop.py           # Animation class (legacy, still supported)
├── contexts.py       # Context-aware rendering (PGA)
├── operations.py     # Geometric operation visualization
├── effects.py        # Visual effects (FadeIn, FadeOut)
├── projection.py     # High-dimensional projection utilities
├── backends/         # Rendering backend abstraction
│   ├── protocol.py   # Backend interface
│   └── pyvista.py    # PyVista/VTK implementation
└── drawing/          # Low-level mesh generation
    └── vectors.py    # Arrow, frame, span meshes
```

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│  High-Level API                                             │
│  Scene (unified), Canvas (static), Animation (legacy)       │
├─────────────────────────────────────────────────────────────┤
│  Backend Abstraction                                        │
│  RenderBackend protocol, PyVistaBackend                     │
├─────────────────────────────────────────────────────────────┤
│  Drawing Primitives                                         │
│  create_frame_mesh(), arrow meshes, span meshes             │
├─────────────────────────────────────────────────────────────┤
│  PyVista / VTK                                              │
│  Plotter, meshes, actors                                    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Scene (`scene.py`) - Recommended

The unified interface for both static and animated visualization. Replaces the need to choose between Canvas and Animation.

```python
from morphis.visuals import Scene, RED, SMALL_SQUARE

# Static display
scene = Scene(theme="obsidian", size=SMALL_SQUARE)
scene.add(v, color=RED)
scene.show()

# Animated display
scene = Scene(theme="obsidian")
scene.add(F, color=RED, filled=True)
scene.fade_in(F, t=0.0, duration=0.5)

for t in times:
    F.data[...] = transform(t)
    scene.capture(t)  # Renders live, syncs to real-time

scene.show()  # Wait for window close
```

**Key methods:**
- `add(element, color, opacity, **kwargs)` - Add element to scene
- `capture(t)` - Render current state at time t (live)
- `show()` - Wait for user to close window
- `fade_in(element, t, duration)` - Schedule fade-in effect
- `fade_out(element, t, duration)` - Schedule fade-out effect

**Design principle:** No data copies stored. Animation happens live during `capture()` calls. Export (to be implemented) will re-run the animation.

### Window Size Constants

Standard sizes for consistent visualization:

```python
from morphis.visuals import SMALL_SQUARE, MEDIUM_RECTANGLE

# Square (1:1)
SMALL_SQUARE = (600, 600)
MEDIUM_SQUARE = (900, 900)
LARGE_SQUARE = (1200, 1200)

# Rectangular (4:3)
SMALL_RECTANGLE = (800, 600)
MEDIUM_RECTANGLE = (1200, 900)
LARGE_RECTANGLE = (1600, 1200)

DEFAULT_SIZE = SMALL_SQUARE
```

### Theme System (`theme.py`)

Four built-in themes:

| Theme    | Background | Character |
|----------|------------|-----------|
| obsidian | Charcoal   | Coral, seafoam, amber |
| paper    | Cream      | Rust, teal, ochre |
| midnight | Near-black | Peach, aqua, gold |
| chalk    | Cool gray  | Crimson, teal, marigold |

Each theme provides:
- Background color (chromatic neutral)
- Axis colors (for coordinate basis)
- Object palette (colors for cycling)

### Backend Abstraction (`backends/`)

Scene uses a pluggable backend system:

```python
class RenderBackend(Protocol):
    def initialize(size, theme, show_basis): ...
    def add_mesh(vertices, faces, color, opacity): ...
    def add_arrows(origins, directions, color, opacity): ...
    def update_mesh(mesh_id, vertices): ...
    def show(interactive): ...
    def wait_for_close(): ...
    def is_closed(): ...
    def process_events(): ...
```

Currently only PyVista backend is implemented. Future backends (Matplotlib, Plotly) can be added.

### Canvas (`canvas.py`) - Low-level

Direct drawing primitives. Use Scene instead for most cases.

```python
canvas = Canvas(theme="obsidian")
canvas.arrow([0, 0, 0], [1, 0, 0], color="red")
canvas.show()
```

### Animation (`loop.py`) - Legacy

The older animation system. Still supported but Scene is preferred.

```python
anim = Animation(theme="obsidian")
anim.watch(F, color=RED)
anim.start()
for t in times:
    F.data[...] = transform(t)
    anim.capture(t)
anim.finish()
```

### PGA Context (`contexts.py`)

Interprets PGA blades as geometric entities:

| Grade | Interpretation | Rendering |
|-------|---------------|-----------|
| 1     | Point/direction | Sphere |
| 2     | Line | Extended segment |
| 3     | Plane | Transparent surface |

### Projection (`projection.py`)

Projects high-dimensional blades to 3D:

```python
scene = Scene(projection=(0, 1, 2))  # Project e1, e2, e3
scene.set_projection((1, 2, 3))      # Switch to e2, e3, e4
```

## Rendering Flow

### Live Animation

```
User code                         Scene
    │                               │
    │  scene.add(element)           │
    ├──────────────────────────────►│ Creates backend visuals
    │                               │
    │  element.data[...] = new      │
    │  scene.capture(t)             │
    ├──────────────────────────────►│ Syncs visuals
    │                               │ Waits for real-time
    │                               │ Processes window events
    │                               │
    │  scene.show()                 │
    ├──────────────────────────────►│ Waits for window close
```

### Static Display

```
scene.add(element)  →  Creates visuals
scene.show()        →  Blocking show, waits for close
```

## Design Decisions

1. **Scene as unified interface** - One class for static and animated

2. **No snapshot storage** - Animation happens live, no data copies

3. **Backend abstraction** - Decouples from PyVista/VTK

4. **Standard window sizes** - Consistent sizing across examples

5. **Real-time sync** - `capture(t)` waits for wall-clock time

6. **Clean window close** - Windows close properly without Ctrl-C

## Extension Points

| To add... | Modify... |
|-----------|-----------|
| New theme | `THEMES` dict in `theme.py` |
| New backend | Implement `RenderBackend` protocol |
| New element type | Add handling in `Scene._create_visuals()` |
| New projection method | `projection.py` |
