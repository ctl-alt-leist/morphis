# Naming Conventions

This document defines the naming conventions used throughout the morphis codebase.

## Guiding Principles

1. **"Vector" means grade-k element**, not specifically grade-1
2. **"Blade" is a property** (`.is_blade`), not a class—a blade is a simple (factorizable) Vector
3. **Consistent terminology**: "grade-k Vector" or "k-vector"
4. **Follow established names** from CLAUDE.md: `lot`, `grade`, `dim`, `metric`
5. **Lot-first layout** for Operators: `(*out_lot, *in_lot, *out_geo, *in_geo)`

---

## Class Names

| Class | Description |
|-------|-------------|
| `Element` | Base class for all GA elements |
| `Tensor` | General (p,q)-tensor, parent of Vector |
| `Vector` | Homogeneous multivector of pure grade k |
| `MultiVector` | General sum of Vectors of different grades |
| `Frame` | Ordered collection of grade-1 Vectors |
| `Metric` | Metric tensor and signature information |
| `Operator` | Linear map between vector spaces |
| `GradedElement` | Base class for elements with grade |
| `CompositeElement` | Base class for elements with multiple grades |

---

## Constructor Functions

### Basis Constructors

| Function | Returns | Description |
|----------|---------|-------------|
| `basis_vector(index, metric)` | Vector | Single basis vector $\mathbf{e}_i$ |
| `basis_vectors(metric)` | tuple[Vector, ...] | All basis vectors $(\mathbf{e}_0, \ldots, \mathbf{e}_{d-1})$ |
| `basis_element(indices, metric)` | Vector | Basis k-vector $\mathbf{e}_{i_1} \wedge \cdots \wedge \mathbf{e}_{i_k}$ |
| `geometric_basis(metric)` | dict[int, tuple] | Complete basis by grade |
| `pseudoscalar(metric)` | Vector | Highest grade basis element $\mathbf{e}_{01\ldots(d-1)}$ |

### Metric Constructors

| Function | Returns | Description |
|----------|---------|-------------|
| `euclidean_metric(dim)` | Metric | Standard Euclidean metric |
| `pga_metric(dim)` | Metric | Projective GA metric (degenerate) |
| `lorentzian_metric(dim)` | Metric | Minkowski spacetime metric |
| `metric(dim, signature, structure)` | Metric | General metric constructor |

---

## Class Properties

### Vector Properties

| Property | Type | Description |
|----------|------|-------------|
| `.grade` | int | The k in k-vector |
| `.dim` | int | Dimension of vector space |
| `.lot` | tuple[int, ...] | Lot (batch) dimensions |
| `.geo` | tuple[int, ...] | Geometric dimensions (depends on grade) |
| `.data` | NDArray | Underlying array |
| `.is_blade` | bool | True if factorizable as $\mathbf{v}_1 \wedge \cdots \wedge \mathbf{v}_k$ |
| `.shape` | tuple | Full data shape = `(*lot, *geo)` |
| `.metric` | Metric | Associated metric |
| `.at` | AtAccessor | Indexing over lot dimensions: `v.at[i]` |
| `.on` | OnAccessor | Indexing over geo dimensions: `v.on[i, j]` |

### Vector Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.form()` | NDArray | Quadratic form (v · v) |
| `.norm()` | NDArray | Norm over geometric dimensions |
| `.unit()` | Vector | Unit vector (norm = 1) |
| `.sum(axis)` | Vector | Sum over specified lot axis |
| `.mean(axis)` | Vector | Mean over specified lot axis |
| `.reverse()` / `.rev()` | Vector | Apply reverse operation |
| `.inverse()` / `.inv()` | Vector | Multiplicative inverse |
| `.conjugate()` / `.conj()` | Vector | Complex conjugation |
| `.hodge()` | Vector | Hodge dual |

### MultiVector Properties

| Property | Type | Description |
|----------|------|-------------|
| `.grades` | tuple[int, ...] | Grades present |
| `.dim` | int | Dimension of vector space |
| `.is_even` | bool | True if only even grades (0, 2, 4, ...) |
| `.is_odd` | bool | True if only odd grades (1, 3, 5, ...) |
| `.is_rotor` | bool | True if even versor with $M\tilde{M} = 1$ |
| `.is_motor` | bool | True if PGA motor |

### Operator Properties

| Property | Type | Description |
|----------|------|-------------|
| `.input_spec` | VectorSpec | Input structure specification |
| `.output_spec` | VectorSpec | Output structure specification |
| `.input_lot` | tuple[int, ...] | Input lot dimensions |
| `.output_lot` | tuple[int, ...] | Output lot dimensions |
| `.input_shape` | tuple | Full input shape = `(*in_lot, *in_geo)` |
| `.output_shape` | tuple | Full output shape = `(*out_lot, *out_geo)` |
| `.shape` | tuple | Data shape = `(*out_lot, *in_lot, *out_geo, *in_geo)` |
| `.is_outermorphism` | bool | True if grade-1 to grade-1 (extendable) |
| `.vector_map` | NDArray | The d×d grade-1 map (if outermorphism) |
| `.H` | Operator | Adjoint (conjugate transpose) |
| `.T` | Operator | Transpose |

**Operator Data Layout (lot-first):**
```
Operator.data.shape = (*out_lot, *in_lot, *out_geo, *in_geo)

Example: scalar→bivector operator with M outputs, N inputs, dim=3
  shape = (M, N, 3, 3)  # lot dims first, then geo dims
```

---

## Operation Functions

### Products (`products.py`)

| Function | Description |
|----------|-------------|
| `wedge(*elements)` | Exterior (wedge) product |
| `antiwedge(*elements)` | Regressive (antiwedge) product |
| `geometric(u, v)` | Geometric (Clifford) product |
| `grade_project(M, k)` | Extract grade-k component |
| `scalar_product(u, v)` | Scalar part of geometric product |
| `commutator(u, v)` | $\frac{1}{2}(uv - vu)$ |
| `anticommutator(u, v)` | $\frac{1}{2}(uv + vu)$ |
| `reverse(u)` | Reverse operator |
| `inverse(u)` | Multiplicative inverse |

### Projections (`projections.py`)

| Function | Description |
|----------|-------------|
| `interior_left(u, v)` | Left contraction $u \lrcorner v$ |
| `interior_right(u, v)` | Right contraction $u \llcorner v$ |
| `interior(u, v)` | Alias for `interior_left` |
| `dot(u, v)` | Inner product (grade-1 vectors) |
| `project(u, v)` | Projection of u onto v |
| `reject(u, v)` | Rejection of u from v |

### Duality (`duality.py`)

| Function | Description |
|----------|-------------|
| `right_complement(u)` | Right complement (metric-independent) |
| `left_complement(u)` | Left complement (metric-independent) |
| `hodge_dual(u)` | Hodge dual (metric-dependent) |

### Norms (`norms.py`)

| Function | Description |
|----------|-------------|
| `form(v)` | Quadratic form (v · v), can be negative |
| `norm(v)` | Norm (sqrt of absolute value of form) |
| `unit(v)` | Unit vector (norm = 1) |
| `conjugate(v)` | Complex conjugation |
| `hermitian_form(v)` | Hermitian quadratic form (for phasors) |
| `hermitian_norm(v)` | Hermitian norm (for phasors) |

### Exponentials (`exponential.py`)

| Function | Description |
|----------|-------------|
| `exp_vector(B)` | Exponential of a vector |
| `log_versor(M)` | Logarithm of a versor |
| `slerp(R0, R1, t)` | Spherical linear interpolation |

### Matrix Representations (`matrix_rep.py`)

| Function | Description |
|----------|-------------|
| `vector_to_array(v)` | Flatten Vector to 1D array |
| `vector_to_vector(arr, grade, metric)` | Reconstruct Vector from array |
| `multivector_to_array(M)` | Flatten MultiVector to array |
| `array_to_multivector(arr, metric)` | Reconstruct MultiVector |
| `left_matrix(A)` | Matrix for left multiplication |
| `right_matrix(A)` | Matrix for right multiplication |
| `operator_to_matrix(L)` | Flatten Operator to 2D matrix |

---

## Method Naming

### Long, Short, and Symbol Forms

| Long Form | Short Form | Symbol | Description |
|-----------|------------|--------|-------------|
| `reverse()` | `rev()` | `~x` | Reverse |
| `inverse()` | `inv()` | `x**(-1)` | Inverse |
| `conjugate()` | `conj()` | — | Complex conjugation |
| `adjoint()` | `adj()` | `.H` | Conjugate transpose |
| `transpose()` | `trans()` | `.T` | Transpose |
| `pseudoinverse()` | `pinv()` | — | Moore-Penrose inverse |

---

## Tensor Contraction

Two APIs for contracting Morphis tensors:

### Bracket Syntax (Preferred)

```python
# String indexing creates IndexedTensor
s = u["a"] * v["a"]           # dot product
outer = u["a"] * v["b"]       # outer product
w = M["ab"] * v["b"]          # matrix-vector
b = G["mnab"] * q["n"]        # batch contraction
```

### Einsum-Style Function

```python
from morphis.algebra import contract

s = contract("a, a ->", u, v)           # dot product
w = contract("ab, b -> a", M, v)        # matrix-vector
result = contract("mn, np, pm ->", A, B, C)  # multi-way
```

The `__getitem__` method dispatches:
- String key → `_index()` → IndexedTensor for contraction
- Other keys → `_slice()` → array slicing

---

## Variable Naming

### Type Conventions

- **Vector variables**: lowercase (`u`, `v`, `w`, `b`)
- **MultiVector variables**: uppercase (`U`, `V`, `W`, `M`)
- **Operator variables**: uppercase (`L`, `G`, `A`)
- **Metric variables**: `m` or `metric`

### Number Suffixes

- Single-letter + number: no underscore (`v1`, `v2`, `R0`, `R1`)
- Word + number: use underscore (`blade_1`, `blade_2`)

### Parameter Names

| Name | Meaning |
|------|---------|
| `lot` | Lot (batch) dimensions tuple, e.g. `lot=(10, 5)` |
| `geo` | Geometric dimensions tuple (derived from grade and dim) |
| `grade` | Grade of k-vector (not `k`) |
| `dim` | Dimension of vector space (not `d`) |
| `metric` | Metric tensor (not `g` or `m`) |

> **Note:** `collection` is deprecated in favor of `lot`. The old syntax `collection=1` still works but emits a deprecation warning. Use `lot=(size,)` instead.

### Why "Lot"?

A **lot** is a group of items considered or processed as a single unit. The term captures three key semantic elements:

1. **Multiplicity** — more than one item
2. **Unity** — treated as a single entity for some purpose
3. **Homogeneity** — items in the lot share common structure

This terminology comes from manufacturing and auction contexts (production lots, auction lots), where items are grouped for batch processing—exactly what lot dimensions represent in morphis.

---

## Docstring Conventions

- Use "Vector" when referring to the class
- Use "grade-k Vector" or "k-vector" for specific grades
- Use "blade" only when referring to simple (factorizable) k-vectors

```python
def example(v: Vector) -> Vector:
    """
    Process a Vector of any grade.

    If the Vector is a blade (simple, factorizable), additional
    optimizations apply.
    """
```
