# Module Structure

This document describes the directory layout for the morphis package.

## Directory Structure

```
src/morphis/
├── __init__.py                  # Main public API
│
├── elements/                    # Mathematical objects (nouns)
│   ├── __init__.py
│   ├── base.py                 # Element, GradedElement, CompositeElement
│   ├── tensor.py               # Tensor base class
│   ├── vector.py               # Vector class (k-vectors)
│   ├── multivector.py          # MultiVector class
│   ├── frame.py                # Frame class
│   ├── metric.py               # Metric, GASignature, GAStructure
│   ├── protocols.py            # Type protocols
│   └── tests/
│
├── operations/                  # Operations on elements (verbs)
│   ├── __init__.py
│   ├── operator.py             # Operator class (linear maps)
│   ├── products.py             # geometric, wedge, antiwedge, reverse, inverse
│   ├── projections.py          # interior products, project, reject, dot
│   ├── duality.py              # Hodge dual, complements
│   ├── norms.py                # form, norm, unit, conjugate
│   ├── exponential.py          # exp_vector, log_versor, slerp
│   ├── factorization.py        # factor, spanning_vectors
│   ├── outermorphism.py        # outermorphism application
│   ├── subspaces.py            # join, meet
│   ├── spectral.py             # eigendecomposition
│   ├── matrix_rep.py           # matrix representations
│   ├── structure.py            # structure constants, einsum signatures
│   ├── _helpers.py             # internal utilities
│   └── tests/
│
├── algebra/                     # Algebraic infrastructure
│   ├── __init__.py
│   ├── specs.py                # VectorSpec
│   ├── patterns.py             # Einsum pattern generation
│   ├── contraction.py          # IndexedTensor, contract()
│   ├── solvers.py              # Linear algebra solvers
│   └── tests/
│
├── transforms/                  # Transformation operations
│   ├── __init__.py
│   ├── actions.py              # rotate, translate, transform
│   ├── projective.py           # PGA utilities
│   ├── rotations.py            # rotor construction
│   └── tests/
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── observer.py             # Observer pattern for tracking objects
│   ├── pretty.py               # Pretty printing utilities
│   └── docgen.py               # API documentation generator
│
├── visuals/                     # Visualization
│   ├── __init__.py
│   ├── canvas.py               # 3D canvas
│   ├── contexts.py             # PGA-specific rendering
│   ├── drawing/
│   │   ├── __init__.py
│   │   └── vectors.py          # Vector mesh generation
│   ├── effects.py              # Visual effects
│   ├── loop.py                 # Animation
│   ├── operations.py           # Operation visualization
│   ├── projection.py           # Dimension projection
│   ├── renderer.py             # Rendering utilities
│   └── theme.py                # Color themes
│
├── examples/                    # Example scripts
│   └── ...
│
└── _legacy/                     # Deprecated code
    └── ...
```

---

## Module Responsibilities

### `elements/`
**Purpose**: Define the mathematical objects (nouns)

| File | Contents |
|------|----------|
| `base.py` | `Element`, `GradedElement`, `CompositeElement` base classes |
| `tensor.py` | `Tensor` class (parent of Vector) |
| `vector.py` | `Vector` class, basis constructors |
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
| `projections.py` | `interior_left()`, `interior_right()`, `project()`, `reject()`, `dot()`, `interior()` |
| `duality.py` | `hodge_dual()`, `right_complement()`, `left_complement()` |
| `norms.py` | `form()`, `norm()`, `unit()`, `conjugate()`, `hermitian_form()`, `hermitian_norm()` |
| `exponential.py` | `exp_vector()`, `log_versor()`, `slerp()` |
| `factorization.py` | `factor()`, `spanning_vectors()` |
| `outermorphism.py` | `apply_exterior_power()`, `apply_outermorphism()` |
| `subspaces.py` | `join()`, `meet()` |
| `spectral.py` | `bivector_to_skew_matrix()`, `bivector_eigendecomposition()`, `principal_vectors()` |
| `matrix_rep.py` | `vector_to_array()`, `vector_to_vector()`, `multivector_to_array()`, `array_to_multivector()`, `left_matrix()`, `right_matrix()`, `operator_to_matrix()` |
| `structure.py` | `permutation_sign()`, `antisymmetrize()`, `levi_civita()`, `generalized_delta()`, einsum signatures |

### `algebra/`
**Purpose**: Algebraic infrastructure for operators

| File | Contents |
|------|----------|
| `specs.py` | `VectorSpec`, `vector_spec()` |
| `patterns.py` | `forward_signature()`, `adjoint_signature()`, `operator_shape()` |
| `contraction.py` | `IndexedTensor`, `contract()` for tensor contraction |
| `solvers.py` | `structured_lstsq()`, `structured_pinv_solve()`, `structured_pinv()`, `structured_svd()` |

### `transforms/`
**Purpose**: Geometric transformation operations

| File | Contents |
|------|----------|
| `rotations.py` | `rotor()`, `rotation_about_point()` |
| `actions.py` | `rotate()`, `translate()`, `transform()` |
| `projective.py` | `point()`, `direction()`, `weight()`, `bulk()`, `euclidean()`, `is_point()`, `is_direction()`, `line()`, `plane()`, `translator()`, `screw()`, distance and incidence functions |

---

## Import Structure

### Top-level API (`morphis/__init__.py`)

```python
# Elements
from morphis.elements import (
    Tensor,
    Vector,
    MultiVector,
    Frame,
    Metric,
    euclidean_metric,
    pga_metric,
    lorentzian_metric,
    basis_vector,
    basis_vectors,
    basis_element,
    geometric_basis,
    pseudoscalar,
)

# Operations
from morphis.operations import (
    Operator,
    geometric,
    wedge,
    antiwedge,
    reverse,
    inverse,
    interior_left,
    interior_right,
    interior,
    project,
    reject,
    dot,
    hodge_dual,
    right_complement,
    left_complement,
    form,
    norm,
    unit,
    conjugate,
    exp_vector,
    log_versor,
    slerp,
)

# Transforms
from morphis.transforms import (
    rotor,
    rotation_about_point,
    rotate,
    translate,
    transform,
)
```

### Module-level imports

```python
# From elements
from morphis.elements import Tensor, Vector, MultiVector, Frame, Metric

# From operations
from morphis.operations import wedge, geometric, reverse, Operator

# From algebra (for advanced users)
from morphis.algebra import VectorSpec
```

---

## Design Decisions

1. **`operations/` not `operators/`**: Operations are verbs (wedge, project), not just the Operator class.

2. **Operator in `operations/`**: The Operator class represents a linear map operation, so it lives with other operations.

3. **`structure.py` in `operations/`**: Einsum signatures and structure constants are implementation details of operations.

4. **Tensor in `elements/`**: Tensor is the mathematical base class for Vector, so it lives with other elements.

5. **Sparse MultiVector**: MultiVector uses `dict[int, Vector]` rather than dense storage—efficient for typical use cases.
