# Class Hierarchy Design

This document describes the architectural relationship between Tensor, Vector, and MultiVector classes in the morphis codebase.

## Current Architecture

```
Element (base: metric, collection)
├── Tensor (data: NDArray, contravariant: int, covariant: int)
│   └── Vector (antisymmetric k-vector, grade = contravariant)
├── GradedElement (data: NDArray, grade: int)
│   └── Frame (ordered collection of grade-1 vectors)
└── CompositeElement (data: dict[int, Vector])
    └── MultiVector
```

**Storage convention**: `(*collection, *geometric)` where geometric = `(dim,) * grade`

---

## Mathematical Context

The mathematical relationships underlying the class hierarchy:

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

Key mathematical facts:
- A **(k,0)-tensor** lives in V^{⊗k}, with components T^{m₁...mₖ}
- An **antisymmetric (k,0)-tensor** lives in ∧ᵏV ⊂ V^{⊗k}
- A **k-vector** in Clifford algebra corresponds to an antisymmetric (k,0)-tensor
- The Clifford algebra Cl(V,g) is ∧V as a vector space, with a different product

---

## Implementation Details

### Tensor Base Class

```python
class Tensor(Element):
    data: NDArray
    contravariant: int  # number of "up" indices
    covariant: int      # number of "down" indices
    metric: Metric
    collection: tuple[int, ...]
```

Storage: `(*collection, *contravariant_dims, *covariant_dims)`

### Vector (k-vector)

```python
class Vector(Tensor):
    grade: int  # = contravariant, covariant always 0
```

- Inherits from Tensor with `contravariant=grade, covariant=0`
- Enforces antisymmetry implicitly through GA operations
- `.is_blade` property checks if factorizable as v₁ ∧ v₂ ∧ ... ∧ vₖ

Storage: `(*collection, dim, dim, ..., dim)` with `grade` copies of `dim`

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
    data: NDArray  # shape: (*collection, span, dim)
    grade: int = 1  # always grade-1 vectors
    span: int  # number of vectors
```

- Efficient storage for ordered collections of grade-1 vectors
- `frame[i]` returns a grade-1 Vector
- Not in Tensor hierarchy (different storage pattern)

---

## Why This Architecture

1. **Mathematical fidelity**: k-vectors ARE antisymmetric (k,0)-tensors
2. **Storage compatibility**: Both use `(*collection, *geometric)` pattern
3. **Minimal overhead**: Tensor base class is thin
4. **Future extensibility**: Can add symmetric tensors, (p,q)-tensors, Forms later
5. **Backward compatibility**: `Blade = Vector` alias maintained

---

## Class Properties

### Vector Properties
- `.is_blade` - True if factorizable as product of grade-1 vectors
- `.grade` - The k in k-vector
- `.dim` - Dimension of underlying vector space
- `.collection` - Batch/collection dimensions

### MultiVector Properties
- `.is_even` - True if only even grades (0, 2, 4, ...)
- `.is_odd` - True if only odd grades (1, 3, 5, ...)
- `.is_rotor` - True if even versor with M * ~M = 1
- `.is_motor` - True if PGA motor (grades {0, 2} with M * ~M = 1)
- `.grades` - List of present grades

---

## Module Organization

| File | Contents |
|------|----------|
| `elements/base.py` | Element, GradedElement, CompositeElement |
| `elements/tensor.py` | Tensor base class |
| `elements/vector.py` | Vector class + basis constructors |
| `elements/multivector.py` | MultiVector class |
| `elements/frame.py` | Frame class |
| `operations/operator.py` | Operator class (linear maps) |
