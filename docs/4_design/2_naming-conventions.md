# Naming Conventions

This document defines the naming conventions used throughout the morphis codebase.

## Guiding Principles

1. **"Vector" means grade-k element**, not specifically grade-1
2. **"Blade" is a property** (`.is_blade`), not a class—a blade is a simple (factorizable) Vector
3. **Consistent terminology**: "grade-k Vector" or "k-vector"
4. **Follow established names** from CLAUDE.md: `collection`, `grade`, `dim`, `metric`

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
| `.collection` | tuple[int, ...] | Batch dimensions |
| `.data` | NDArray | Underlying array |
| `.is_blade` | bool | True if factorizable as $\mathbf{v}_1 \wedge \cdots \wedge \mathbf{v}_k$ |
| `.shape` | tuple | Full data shape |
| `.metric` | Metric | Associated metric |

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
| `.input_spec` | VectorSpec | Input structure |
| `.output_spec` | VectorSpec | Output structure |
| `.is_outermorphism` | bool | True if grade-1 to grade-1 (extendable) |
| `.vector_map` | NDArray | The d×d grade-1 map (if outermorphism) |
| `.H` | Operator | Adjoint (conjugate transpose) |
| `.T` | Operator | Transpose |

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
| `norm_squared(u)` | Squared norm (can be negative) |
| `norm(u)` | Norm (sqrt of absolute value) |
| `normalize(u)` | Normalize to unit norm |
| `conjugate(u)` | Complex conjugation |
| `hermitian_norm(u)` | Hermitian norm (for phasors) |
| `hermitian_norm_squared(u)` | Hermitian squared norm |

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
| `collection` | Batch/collection dimensions (not `batch_dims`) |
| `grade` | Grade of k-vector (not `k`) |
| `dim` | Dimension of vector space (not `d`) |
| `metric` | Metric tensor (not `g` or `m`) |

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
