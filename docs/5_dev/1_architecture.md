# Architecture

This document describes the architectural design of morphis, including design philosophy, class relationships, module responsibilities, and key design decisions.

## Design Philosophy

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

For k-vectors of arbitrary grade, the number of indices varies with k. This requires inspecting grade to build einsum strings and applying 1/k! normalization. Morphis accepts this limitation while maximizing einsum use within each grade and keeping grade-dispatch logic minimal and isolated.

### Geometric Algebra as Foundation

Every geometric structure—from vectors to complex manifolds—is expressed through geometric algebra:
- Unified operations (wedge, geometric product, duality)
- Coordinate-free geometric meaning
- Efficient einsum-based implementation
- Natural extension to different signatures and dimensions

### Mathematics, Not Applications

The package provides geometric and algebraic tools. Specific applications live in examples.

**In package:** Vectors (k-vectors), multivectors, metrics, products, norms, projections, rotors, translators, motors, PGA operations.

**In examples:** 3D/4D animations, specific geometric constructions.

### No Premature Abstraction

Create modules when duplication emerges, not preemptively. Let actual usage patterns drive structure.

## Class Hierarchy

```
Element (base: metric, lot)
├── Tensor (data: NDArray, contravariant: int, covariant: int)
│   └── Vector (antisymmetric k-vector, grade = contravariant)
├── GradedElement (data: NDArray, grade: int)
│   └── Frame (ordered set of grade-1 vectors)
└── CompositeElement (data: dict[int, Vector])
    └── MultiVector
```

**Storage convention**: `(*lot, *geo)` where geo = `(dim,) * grade`

## Mathematical Context

The mathematical relationships underlying the design:

```
Tensor Algebra T(V)              Clifford Algebra Cl(V,g)
       │                                │
       │ antisymmetrize                 │ Z-grading
       ▼                                ▼
Exterior Algebra ∧V  ════════>  Graded Cl(V,g)
                                     │
                                     ▼
                              Objects in Cl(V,g):
                              ├── MultiVector (general)
                              └── Vector (homogeneous grade-k)
```

Key facts:
- A **(k,0)-tensor** lives in V^{⊗k}, with components T^{m₁...mₖ}
- An **antisymmetric (k,0)-tensor** lives in ∧ᵏV ⊂ V^{⊗k}
- A **k-vector** in Clifford algebra corresponds to an antisymmetric (k,0)-tensor
- The Clifford algebra Cl(V,g) is ∧V as a vector space, with a different product

## Class Details

### Tensor Base Class

```python
class Tensor(Element):
    data: NDArray
    contravariant: int  # number of "up" indices
    covariant: int      # number of "down" indices
    metric: Metric
    lot: tuple[int, ...]
```

Storage: `(*lot, *contravariant_dims, *covariant_dims)`

### Vector (k-vector)

```python
class Vector(Tensor):
    grade: int  # = contravariant, covariant always 0
```

- Inherits from Tensor with `contravariant=grade, covariant=0`
- Enforces antisymmetry implicitly through GA operations
- `.is_blade` property checks if factorizable as v₁ ∧ v₂ ∧ ... ∧ vₖ

Storage: `(*lot, dim, dim, ..., dim)` with `grade` copies of `dim`

### MultiVector

```python
class MultiVector(CompositeElement):
    data: dict[int, Vector]  # grade -> Vector
```

- Sparse representation: only stores non-zero grades
- Each component is a homogeneous Vector

### Frame

```python
class Frame(GradedElement):
    data: NDArray  # shape: (*lot, span, dim)
    grade: int = 1  # always grade-1 vectors
    span: int  # number of vectors
```

- Efficient storage for ordered sets of grade-1 vectors
- `frame[i]` returns a grade-1 Vector
- Not in Tensor hierarchy (different storage pattern)

## Module Responsibilities

### `elements/`
**Purpose**: Define the mathematical objects (nouns)

| File | Contents |
|------|----------|
| `base.py` | `Element`, `GradedElement`, `CompositeElement` base classes |
| `tensor.py` | `Tensor` base class |
| `vector.py` | `Vector` class + basis constructors |
| `multivector.py` | `MultiVector` class |
| `frame.py` | `Frame` class |
| `metric.py` | `Metric`, `GASignature`, `GAStructure`, factory functions |
| `protocols.py` | `Graded`, `Spanning`, `Transformable` protocols |

### `operations/`
**Purpose**: Operations on elements (verbs), including Operator class

| File | Contents |
|------|----------|
| `operator.py` | `Operator` class for linear maps |
| `products.py` | `geometric()`, `wedge()`, `antiwedge()`, `reverse()`, `inverse()` |
| `projections.py` | `interior_left()`, `interior_right()`, `project()`, `reject()`, `dot()` |
| `duality.py` | `hodge_dual()`, `right_complement()`, `left_complement()` |
| `norms.py` | `form()`, `norm()`, `unit()`, `conjugate()` |
| `exponential.py` | `exp_vector()`, `log_versor()`, `slerp()` |
| `factorization.py` | `factor()`, `spanning_vectors()` |
| `outermorphism.py` | `apply_exterior_power()`, `apply_outermorphism()` |
| `subspaces.py` | `join()`, `meet()` |
| `spectral.py` | `bivector_eigendecomposition()` |
| `matrix_rep.py` | `vector_to_array()`, `left_matrix()`, etc. |
| `structure.py` | `generalized_delta()`, `levi_civita()`, einsum signatures |

### `algebra/`
**Purpose**: Algebraic infrastructure

| File | Contents |
|------|----------|
| `specs.py` | `VectorSpec` |
| `patterns.py` | Einsum pattern generation |
| `solvers.py` | Linear algebra solvers |

### `transforms/`
**Purpose**: Geometric transformation operations

| File | Contents |
|------|----------|
| `rotations.py` | `rotor()` construction |
| `actions.py` | `rotate()`, `translate()`, `transform()` |
| `projective.py` | PGA utilities: `point()`, `direction()`, `line()`, etc. |

## Design Decisions

### 1. Three-Layer API Naming

Mathematical operations follow a consistent naming pattern with three forms:

| Long Form | Short Form | Symbol Form | Description |
|-----------|------------|-------------|-------------|
| `reverse()` | `rev()` | `~x` | Reverses blade factor order |
| `inverse()` | `inv()` | `x**(-1)` | Multiplicative inverse |
| `conjugate()` | `conj()` | — | Complex conjugation |
| `adjoint()` | `adj()` | `.H` | Conjugate transpose (Operator) |
| `transpose()` | `trans()` | `.T` | Transpose (Operator) |
| `pseudoinverse()` | `pinv()` | — | Moore-Penrose inverse (Operator) |

### 2. Vector = k-vector
The `Vector` class represents any grade-k homogeneous multivector. The term "blade" becomes a property (`.is_blade`), not a class. This naming emphasizes that all grades are vectors in their respective exterior power spaces ∧ᵏV.

### 2. Full antisymmetric tensor storage
Vectors store full `(dim,) * grade` arrays rather than compressed unique components. This allows direct einsum operations without index translation, at the cost of redundant storage for the antisymmetric parts.

### 3. Operator in operations/
The `Operator` class lives in `operations/` rather than `elements/` because it represents an operation (linear map) rather than a pure mathematical object.

### 4. Sparse MultiVector
`MultiVector` uses a dictionary `{grade: Vector}` rather than a dense array over all grades. This is efficient for typical use cases (rotors have grades {0, 2}, motors have grades {0, 2}).

### 5. Lot dimensions
All objects support leading "lot" dimensions for batch processing. Operations use `...` in einsum patterns and `[..., idx]` for slicing to handle arbitrary batch shapes. The storage convention is `(*lot, *geo)` for all elements.

### 6. Metric as parameter
Metric-dependent operations take an optional metric parameter rather than relying on global state:

```python
norm(b, metric=g)
interior(u, v, metric=g)
```

### 7. Element multiplication semantics

The four core elements (Vector, Frame, MultiVector, Operator) have well-defined multiplication:

| Left | Right | Result | Operation |
|------|-------|--------|-----------|
| Vector | Vector | MultiVector | Geometric product |
| Vector | MultiVector | MultiVector | Geometric product |
| MultiVector | MultiVector | MultiVector | Geometric product |
| scalar | any | same type | Scalar multiplication |
| Operator | Vector | Vector | Apply operator (or exterior power) |
| Operator | Frame | Frame | Apply to each vector |
| Operator | MultiVector | MultiVector | Outermorphism (grade-1→grade-1 only) |
| Operator | Operator | Operator | Composition |

## Naming Conventions

### Variable Names
- `lot` — not `collection`, `collection_dims`, `batch_dims`
- `geo` — not `geometric`, `geometric_shape`
- `grade` — not `k`, `blade_grade`
- `dim` — not `d`, `dimension`, `ndim`
- `metric` — not `g`, `m`

> **Note:** `collection` is deprecated in favor of `lot`. The old parameter still works but emits a deprecation warning.

### Type Hints
- Vector variables: lowercase (`u, v, w, b`)
- MultiVector variables: uppercase (`U, V, W, M`)
- Single-letter + number: no underscore (`v1, v2`)
- Word + number: use underscore (`blade_1, blade_2`)

### Internal Helpers
Private functions use underscore prefix and explicit type suffixes:
- `_wedge_vec_vec()` — Vector × Vector
- `_wedge_vec_mv()` — Vector × MultiVector
- `_geometric_mv_mv()` — MultiVector × MultiVector
