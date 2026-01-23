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
│   ├── blade.py       # Blade class (k-vectors)
│   ├── multivector.py # MultiVector class (sums of blades)
│   ├── metric.py      # Metric definitions (Euclidean, PGA, etc.)
│   ├── frame.py       # Ordered vector collections
│   ├── operator.py    # Linear maps between blade spaces
│   └── elements.py    # GradedElement, CompositeElement base classes
│
├── algebra/        # Linear algebra for operators
│   ├── specs.py       # BladeSpec for operator I/O structure
│   ├── patterns.py    # Einsum signature generation
│   └── solvers.py     # SVD, pseudoinverse, least squares
│
├── operations/     # GA operations
│   ├── products.py       # Wedge, interior, geometric products
│   ├── norms.py          # Norm, normalize
│   ├── duality.py        # Complement, Hodge dual
│   ├── projections.py    # Project, reject
│   ├── subspaces.py      # Join, meet
│   ├── structure.py      # Einsum signatures, Levi-Civita
│   ├── factorization.py  # Blade factorization
│   └── outermorphism.py  # Exterior power, outermorphism application
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

### 1. Three-Layer API Naming Convention

Mathematical operations follow a consistent naming pattern with three forms:

| Long Form | Short Form | Symbol Form | Description |
|-----------|------------|-------------|-------------|
| `reverse()` | `rev()` | `~x` | Reverses blade factor order |
| `inverse()` | `inv()` | `x**(-1)` | Multiplicative inverse |
| `conjugate()` | `conj()` | — | Complex conjugation |
| `adjoint()` | `adj()` | `.H` | Conjugate transpose (Operator) |
| `transpose()` | `trans()` | `.T` | Transpose (Operator) |
| `pseudoinverse()` | `pinv()` | — | Moore-Penrose inverse (Operator) |

This allows users to choose the form that best fits their context:
- Long form for clarity in documentation and teaching
- Short form for concise mathematical expressions
- Symbol form when it matches standard notation

### 2. Full Antisymmetric Tensor Storage

Blades store all $d^k$ components, not just independent ones. This enables:
- Direct einsum operations without index bookkeeping
- Uniform batch dimension handling via `...`
- Simple grade-agnostic algorithms

### 3. Collection Dimensions

All operations support leading batch dimensions:

```python
# Shape: (batch, d) for vectors
# Shape: (batch, d, d) for bivectors
# Einsum ... absorbs batch dimensions automatically
```

### 4. Metric as Parameter

Metric-dependent operations take an optional metric parameter rather than relying on global state:

```python
norm(blade, metric=g)
interior(u, v, metric=g)
```

### 5. Separation of Animation and Rendering

The animation system separates transformation logic from rendering:
- External code owns and modifies transforms
- Canvas reads transforms and renders
- No bidirectional coupling

### 6. Linear Operators

The `Operator` class represents structured linear maps between blade spaces. Unlike matrix representations, operators maintain geometric structure throughout:

```python
from morphis.elements import Operator, Blade, euclidean
from morphis.algebra import BladeSpec

# Define transfer operator G mapping scalars to bivectors
G = Operator(
    data=G_data,  # shape: (d, d, M, N) for scalar->bivector
    input_spec=BladeSpec(grade=0, collection_dims=1, dim=d),
    output_spec=BladeSpec(grade=2, collection_dims=1, dim=d),
    metric=euclidean(d),
)

# Forward application: B = G * I
B = G * I

# Inverse operations
x = G.solve(B)          # Least squares
G_inv = G.pinv()        # Pseudoinverse
U, S, Vt = G.svd()      # Structured SVD
```

Key properties:
- `input_collection`: Shape of input collection dimensions
- `output_collection`: Shape of output collection dimensions
- `input_shape`: Full shape expected for input blade data
- `output_shape`: Full shape of output blade data

### 7. Outermorphisms

An **outermorphism** is a linear map that preserves the wedge product structure:

$$f(a \wedge b) = f(a) \wedge f(b)$$

Key insight: An outermorphism is completely determined by its action on grade-1 (vectors). The extension to grade-$k$ is the $k$-th exterior power $\bigwedge^k A$.

In Morphis, any `Operator` mapping grade-1 → grade-1 can act as an outermorphism:

```python
# Create a rotation matrix as an operator
R = Operator(
    data=rotation_matrix,  # shape: (d, d)
    input_spec=BladeSpec(grade=1, collection=0, dim=d),
    output_spec=BladeSpec(grade=1, collection=0, dim=d),
    metric=euclidean(d),
)

# Check if operator can extend to all grades
R.is_outermorphism  # True

# Apply to vector (direct application)
v_rotated = R * v

# Apply to bivector (uses 2nd exterior power automatically)
B_rotated = R * B

# Apply to full multivector (extends to all grades)
M_rotated = R * M
```

The exterior power is computed on-demand via einsum: $k$ copies of the $d \times d$ vector map contract with the $k$ indices of the blade. This preserves the wedge product:

$$(\bigwedge^k A)(v_1 \wedge \cdots \wedge v_k) = A(v_1) \wedge \cdots \wedge A(v_k)$$

Key property: The action on the pseudoscalar equals multiplication by the determinant:

$$(\bigwedge^d A)(\mathbb{1}) = \det(A) \cdot \mathbb{1}$$

### 8. Element Operations

The four core elements (Blade, Frame, MultiVector, Operator) have well-defined multiplication semantics:

| Left | Right | Result | Operation |
|------|-------|--------|-----------|
| Blade | Blade | MultiVector | Geometric product |
| Blade | MultiVector | MultiVector | Geometric product |
| MultiVector | Blade | MultiVector | Geometric product |
| MultiVector | MultiVector | MultiVector | Geometric product |
| scalar | any | same type | Scalar multiplication |
| Operator | Blade | Blade | Apply operator (or exterior power for outermorphisms) |
| Operator | Frame | Frame | Apply to each vector |
| Operator | MultiVector | MultiVector | Outermorphism (grade-1→grade-1 operators only) |
| Operator | Operator | Operator | Composition |

Some operations are explicitly not supported (raise `TypeError`):
- `Blade * Operator` — use `Operator * Blade`
- `Frame * Operator` — use `Operator * Frame`
- `Operator * MultiVector` for non-outermorphism operators — only grade-1→grade-1 can extend
- `Frame ^ anything` — wedge with Frame not yet implemented

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
