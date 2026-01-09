# Visualization Architecture

A bird's-eye view of the `morphis.visualization` subpackage.

## Module Structure

```
src/morphis/visualization/
├── __init__.py       # Public API exports
├── theme.py          # Color themes and palettes
├── canvas.py         # Base 3D rendering canvas
├── drawing.py        # Low-level blade drawing primitives
├── blades.py         # Grade-specific blade visualization
├── contexts.py       # Context-aware rendering (PGA)
├── operations.py     # Geometric operation visualization
├── animated.py       # AnimatedCanvas for real-time updates
├── transforms.py     # Animation transforms and easing
└── projection.py     # High-dimensional projection utilities
```

## Layer Overview

The package is organized in layers from low-level to high-level:

```
┌─────────────────────────────────────────────────────────────┐
│  High-Level API                                             │
│  visualize_blade(), visualize_blades(), visualize_pga_*()   │
├─────────────────────────────────────────────────────────────┤
│  Grade/Context Renderers                                    │
│  render_vector(), render_bivector(), render_pga_line()      │
├─────────────────────────────────────────────────────────────┤
│  Canvas Abstraction                                         │
│  Canvas.arrow(), Canvas.plane(), Canvas.curve()             │
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
- `arrow(start, end)` - Vector arrows with configurable proportions
- `curve(points)` - Smooth splines rendered as tubes
- `point(position)` - Spheres
- `plane(origin, normal)` - Semi-transparent quads
- `basis_vectors()` - Coordinate axes with labels

**Features:**
- Automatic color cycling through theme palette
- Camera control (position, focal point)
- Screenshot export

### Theme System (`theme.py`)

Four built-in themes with color palettes based on color theory principles:

| Theme    | Background | Character |
|----------|------------|-----------|
| obsidian | Charcoal   | Coral, seafoam, amber |
| paper    | Cream      | Rust, teal, ochre |
| midnight | Near-black | Peach, aqua, gold |
| chalk    | Cool gray  | Crimson, teal, marigold |

Each theme provides:
- Background color (chromatic neutral, never pure black/white)
- Basis colors (e1, e2, e3 with distinct hues)
- Object palette (6-8 colors for automatic cycling)
- Accent and muted colors

### Grade-Specific Rendering (`blades.py`)

Each grade has dedicated visualization:

**Scalars (grade 0):** Sphere at origin, radius = magnitude

**Vectors (grade 1):** Arrow from origin

**Bivectors (grade 2):** Four modes:
- `circle` - Circle in plane, radius proportional to sqrt(magnitude)
- `parallelogram` - Two spanning vectors with edges and surface
- `plane` - Semi-transparent infinite plane with normal arrow
- `circular_arrow` - Circle with orientation indicator

**Trivectors (grade 3):** Two modes:
- `parallelepiped` - Three spanning vectors forming a box
- `sphere` - Sphere with radius proportional to cube root of magnitude

### PGA Context (`contexts.py`)

Interprets PGA blades as geometric entities:

| Grade | Interpretation | Rendering |
|-------|---------------|-----------|
| 1     | Point or direction | Sphere at Euclidean location |
| 2     | Line | Extended line segment |
| 3     | Plane | Semi-transparent infinite plane |

```python
p = point([1, 2, 0])
l = line(p, point([3, 4, 0]))
visualize_pga_scene(canvas, [p, l])
```

## Rendering Pipelines

### Static Rendering

```
Blade
  │
  ▼
visualize_blade()
  │
  ├── [if dim > 3] → project_blade()
  │
  ├── [grade dispatch]
  ▼
render_vector() / render_bivector() / ...
  │
  ▼
Canvas.arrow() / Canvas.plane() / ...
  │
  ▼
PyVista mesh creation
  │
  ▼
show()
```

### Animated Rendering

The animation system separates transformation logic from rendering:

```
External code                    AnimatedCanvas
     │                                │
     │  blade.visual_transform        │
     │ ◄──────────────────────────────┤ track(blade)
     │                                │
     ▼                                │
rotate_blade(blade, ...)              │
translate_blade(blade, ...)           │
     │                                │
     │  blade.visual_transform        │
     ├───────────────────────────────►│ update()
     │                                │
     │                                ▼
     │                          sync VTK transforms
     │                                │
     │                                ▼
     │                          render frame
```

**Key principle:** AnimatedCanvas is a pure renderer. It reads `blade.visual_transform` but never modifies it. External code owns transformation logic.

#### Object Tracking

When `track(blade)` is called, the canvas creates a `TrackedBlade` record:

```python
class TrackedBlade:
    blade_ref: weakref      # Weak reference to avoid preventing GC
    edges_mesh: pv.PolyData # PyVista mesh for edges
    faces_mesh: pv.PolyData # PyVista mesh for faces
    edges_actor: vtkActor   # VTK actor for edges
    faces_actor: vtkActor   # VTK actor for faces
    vtk_transform: vtkTransform  # Shared transform applied to both actors
    color: Color
```

**Tracking flow:**

1. **Geometry creation** - A unit geometry is created (e.g., unit parallelepiped for trivectors with corners at origin and basis vectors)

2. **Mesh to actor** - Meshes are added to the PyVista plotter, returning VTK actors

3. **Transform binding** - A `vtkTransform` is created and bound to both actors via `SetUserTransform()`. This means the same transform affects both edges and faces.

4. **Weak reference** - A `weakref` to the blade is stored, allowing the blade to be garbage collected if no other references exist

5. **Storage** - The `TrackedBlade` is stored in `_tracked` dict, keyed by `id(blade)`

**Update cycle:**

```python
def update(self):
    for tracked in self._tracked.values():
        tracked.sync_from_blade()  # Read transform from blade
    self._render()                 # Render the frame
```

The `sync_from_blade()` method:
1. Gets the blade via the weak reference
2. Reads `blade.visual_transform` (rotation matrix + translation vector)
3. Converts to VTK transform (rotation via axis-angle, then translation)
4. The actors automatically use the updated transform on next render

**Why weak references?** If external code drops all references to a blade, the blade can be garbage collected. The canvas will find `blade_ref()` returns `None` and skip that tracked object. This prevents the canvas from keeping blades alive indefinitely.

### Animation Sequences (`transforms.py`)

Fluent API for building animations:

```python
seq = AnimationSequence()
seq.rotate(angle=2*pi, axis=[0, 0, 1], duration=4.0)
   .translate(target=[1, 1, 0], duration=2.0)
   .wait(1.0)

# In animation loop:
transform = seq.evaluate(current_time)
blade.visual_transform.rotation = transform.rotation
canvas.update()
```

**Easing functions:** linear, cubic, sine, quadratic (in/out variants)

## Blade Factorization

Bivectors and trivectors are factored into spanning vectors for visualization:

**Bivector factorization:** SVD-based decomposition finds two orthogonal vectors spanning the plane.

**Trivector factorization:** Extracts three spanning vectors from the antisymmetric tensor.

This enables rendering abstract k-blades as concrete geometric shapes.

## Projection (`projection.py`)

Projects high-dimensional blades to 3D for visualization:

```python
config = ProjectionConfig(method="principal", target_dim=3)
projected = project_blade(blade_4d, config)
```

**Methods:**
- `slice` - Select fixed axis indices
- `principal` - Select axes with largest component magnitudes

## Style Configuration

Dataclasses separate style from rendering:

```python
style = BladeStyle(
    vector_color=(1, 0, 0),
    bivector_mode="parallelogram",
    trivector_opacity=0.3,
)
visualize_blade(canvas, blade, style=style)
```

**Style classes:**
- `BladeStyle` - General blade visualization
- `PGAStyle` - Extends with PGA-specific parameters
- `OperationStyle` - For meet/join visualization

## Operation Visualization (`operations.py`)

Visualizes geometric algebra operations:

```python
render_meet_join(canvas, plane1, plane2)  # Shows intersection and union
render_with_dual(canvas, bivector)         # Shows blade and its dual
```

Inputs rendered muted, results highlighted.

## Extension Points

| To add... | Modify... |
|-----------|-----------|
| New theme | `THEMES` dict in `theme.py` |
| New grade visualization | `blades.py` render functions |
| New context (e.g., CGA) | `contexts.py` |
| New animation type | `AnimationSequence` methods |
| New projection method | `ProjectionConfig.method` |

## Key Design Decisions

1. **Canvas hides PyVista** - Users work with geometric primitives, not meshes

2. **Grade-based dispatch** - Each grade has dedicated rendering, enabling grade-specific modes

3. **Separation of animation and rendering** - AnimatedCanvas reads transforms, external code writes them

4. **Style as configuration** - Styles are data, not behavior; easy to share and serialize

5. **Projection as middleware** - Applied transparently when dim > 3

6. **Color cycling** - Automatic palette cycling for multi-object scenes without manual color assignment
