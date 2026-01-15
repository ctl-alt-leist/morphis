# Morphis Project Overview

A unified mathematical framework for geometric computation.

## Vision

**Morphis** provides elegant tools for working with geometric algebra, manifolds, and their applications across mathematics and physics.

The name derives from Greek *morphe* (form)—embodying the transformation and adaptation of geometric structures across different contexts while preserving their essential nature.

## Core Philosophy

### Manifest Generality

Code should be the mathematical expression it implements. The generality should be visible in the structure, not buried in conditional branches.

```python
# Yes: manifest generality via einsum
einsum("ab, ...a, ...b -> ...", g, u, v)

# No: conditional logic about mathematics
if u.shape == (d,):
    result = ...
elif u.shape == (n, d):
    result = ...
```

### Geometric Algebra as Foundation

Every geometric structure—from vectors to complex manifolds—is expressed through geometric algebra:
- Unified operations (wedge, geometric product, duality)
- Coordinate-free geometric meaning
- Efficient einsum-based implementation
- Natural extension to different signatures and dimensions

### Mathematics, Not Applications

The package provides geometric and algebraic tools. Specific applications live in examples.

**In package:**
- Blades, multivectors, metrics
- Products, norms, projections
- Rotors, translators, motors
- PGA operations (points, lines, planes)

**In examples:**
- 3D/4D animations
- Specific geometric constructions

### No Premature Abstraction

Create modules when duplication emerges, not preemptively. Let actual usage patterns drive structure.

## Package Structure

```
morphis/
├── elements/       # Core GA objects
│   ├── blade.py       # Blade class
│   ├── multivector.py # MultiVector class
│   ├── metric.py      # Metric definitions
│   ├── frame.py       # Orthonormal frames
│   └── elements.py    # GradedElement, CompositeElement base classes
│
├── operations/     # GA operations
│   ├── products.py    # Wedge, interior, geometric products
│   ├── norms.py       # Norm, normalize
│   ├── duality.py     # Complement, Hodge dual
│   ├── projections.py # Project, reject
│   ├── subspaces.py   # Join, meet
│   ├── structure.py   # Einsum signatures, Levi-Civita
│   └── factorization.py
│
├── transforms/     # Transformations
│   ├── rotations.py   # Rotor construction and application
│   ├── actions.py     # Sandwich products
│   └── projective.py  # PGA: points, lines, planes, distances
│
├── visuals/        # Visualization
│   ├── canvas.py      # Base 3D rendering
│   ├── renderer.py    # Blade rendering
│   ├── loop.py        # Animation loop
│   ├── theme.py       # Color themes
│   └── drawing/       # Mesh generation
│
├── utils/          # Utilities
│   ├── easing.py      # Animation easing functions
│   ├── observer.py    # Observer pattern
│   └── pretty.py      # Pretty printing
│
└── examples/       # Runnable demos
```

## Key Design Decisions

### 1. Full Antisymmetric Tensor Storage

Blades store all $d^k$ components, not just independent ones. This enables:
- Direct einsum operations without index bookkeeping
- Uniform batch dimension handling via `...`
- Simple grade-agnostic algorithms

### 2. Collection Dimensions

All operations support leading batch dimensions:

```python
# Shape: (batch, d) for vectors
# Shape: (batch, d, d) for bivectors
# Einsum ... absorbs batch dimensions automatically
```

### 3. Metric as Parameter

Metric-dependent operations take an optional metric parameter rather than relying on global state:

```python
norm(blade, metric=g)
interior(u, v, metric=g)
```

### 4. Separation of Animation and Rendering

The animation system separates transformation logic from rendering:
- External code owns and modifies transforms
- Canvas reads transforms and renders
- No bidirectional coupling

## The Manifest Generality Challenge

### What Einsum Achieves

For vectors, einsum provides manifest generality:

```python
def dot(u, v, g):
    return einsum("ab, ...a, ...b -> ...", g, u, v)
```

One line handles all batch shapes. The code is the mathematical expression.

### The k-Blade Problem

For blades of arbitrary grade, the number of indices varies with $k$. This requires:
- Inspecting grade to build einsum strings
- Applying $1/k!$ normalization

The code now contains logic *about* the mathematics rather than *being* the mathematics.

### Resolution

Morphis accepts this limitation for grade-dependent operations while:
- Maximizing einsum use within each grade
- Caching generated einsum signatures
- Keeping the grade-dispatch logic minimal and isolated

## Resources

### Mathematical Background

- *Geometric Algebra for Computer Science* (Dorst, Fontijne, Mann)
- *Geometric Algebra for Physicists* (Doran, Lasenby)
- *Projective Geometric Algebra* (Gunn)

### Implementation References

- ganja.js (JavaScript GA visualization)
- clifford (Python Clifford algebra library)
