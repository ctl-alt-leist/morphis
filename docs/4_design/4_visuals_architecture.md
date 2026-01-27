# Visualization Architecture

A bird's-eye view of the `morphis.visuals` subpackage.

## Module Structure

```
src/morphis/visuals/
├── __init__.py       # Public API exports
├── theme.py          # Color themes and palettes
├── canvas.py         # Base 3D rendering canvas
├── renderer.py       # Blade rendering
├── contexts.py       # Context-aware rendering (PGA)
├── operations.py     # Geometric operation visualization
├── loop.py           # Animation loop and timeline
├── effects.py        # Visual effects
├── projection.py     # High-dimensional projection utilities
└── drawing/          # Low-level mesh generation
    ├── vectors.py    # Arrow meshes
    ├── bivectors.py  # Plane/parallelogram meshes
    └── trivectors.py # Volume meshes
```

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│  High-Level API                                             │
│  Canvas, AnimationLoop, render functions                    │
├─────────────────────────────────────────────────────────────┤
│  Grade/Context Renderers                                    │
│  render_vector(), render_bivector(), render_pga_*()         │
├─────────────────────────────────────────────────────────────┤
│  Drawing Primitives                                         │
│  arrow_mesh(), plane_mesh(), parallelepiped_mesh()          │
├─────────────────────────────────────────────────────────────┤
│  PyVista / VTK                                              │
│  Plotter, meshes, actors                                    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Canvas (`canvas.py`)

The main rendering surface. Wraps PyVista with a geometry-focused API.

```python
canvas = Canvas(theme="obsidian")
canvas.arrow([0, 0, 0], [1, 0, 0], color="red")
canvas.plane([0, 0, 0], [0, 0, 1])
canvas.basis_vectors()
canvas.show()
```

**Primitives:**
- `arrow(start, end)` - Vector arrows
- `curve(points)` - Smooth splines
- `point(position)` - Spheres
- `plane(origin, normal)` - Semi-transparent quads
- `basis_vectors()` - Coordinate axes

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
- Basis colors (e1, e2, e3)
- Object palette (6-8 colors for cycling)

### Renderer (`renderer.py`)

Grade-specific blade visualization:

**Scalars (grade 0):** Sphere at origin

**Vectors (grade 1):** Arrow from origin

**Bivectors (grade 2):** Multiple modes:
- `circle` - Circle in plane
- `parallelogram` - Two spanning vectors
- `plane` - Semi-transparent surface

**Trivectors (grade 3):**
- `parallelepiped` - Three spanning vectors
- `sphere` - Volume indicator

### Animation Loop (`loop.py`)

Timeline-based animation system:

```python
loop = AnimationLoop(canvas)
loop.add_timeline(blade, timeline)
loop.run()
```

Key principle: The loop reads transforms from blades but never modifies them. External code owns transformation logic.

### PGA Context (`contexts.py`)

Interprets PGA blades as geometric entities:

| Grade | Interpretation | Rendering |
|-------|---------------|-----------|
| 1     | Point/direction | Sphere |
| 2     | Line | Extended segment |
| 3     | Plane | Transparent surface |

## Rendering Pipelines

### Static Rendering

```
Blade
  │
  ▼
render_blade()
  │
  ├── [if dim > 3] → project_blade()
  │
  ├── [grade dispatch]
  ▼
render_vector() / render_bivector() / ...
  │
  ▼
Canvas methods
  │
  ▼
PyVista mesh
  │
  ▼
show()
```

### Animated Rendering

```
External code                    AnimationLoop
     │                                │
     │  blade transforms              │
     │ ◄──────────────────────────────┤ track(blade)
     │                                │
     ▼                                │
apply_rotor(blade, ...)               │
translate(blade, ...)                 │
     │                                │
     │  blade transforms              │
     ├───────────────────────────────►│ update()
     │                                │
     │                                ▼
     │                          sync VTK transforms
     │                                │
     │                                ▼
     │                          render frame
```

## Projection (`projection.py`)

Projects high-dimensional blades to 3D:

```python
projected = project_blade(blade_4d, target_dim=3)
```

Methods:
- `slice` - Select fixed axis indices
- `principal` - Select axes with largest components

## Design Decisions

1. **Canvas hides PyVista** - Users work with geometric primitives

2. **Grade-based dispatch** - Each grade has dedicated rendering

3. **Separation of animation and rendering** - Loop reads transforms, external code writes

4. **Style as configuration** - Styles are data, easy to share

5. **Automatic color cycling** - Multi-object scenes without manual assignment

## Extension Points

| To add... | Modify... |
|-----------|-----------|
| New theme | `THEMES` dict in `theme.py` |
| New grade visualization | `renderer.py` |
| New context (e.g., CGA) | `contexts.py` |
| New projection method | `projection.py` |
