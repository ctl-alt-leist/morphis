# API Overview

This document provides a practical guide to the morphis API, focusing on operator syntax, method shortcuts, and common usage patterns.

## Core Concepts

### Lot and Geo

Every `Vector` has two kinds of dimensions:

- **lot** (layout): Batch/collection dimensions for operating on many vectors at once
- **geo** (geometric): Dimensions representing the k-vector structure (shape = `dim^grade`)

```python
# Vector with lot=(10, 5) representing 50 grade-1 vectors
v = Vector(data, grade=1, metric=g, lot=(10, 5))
v.lot   # (10, 5)
v.geo   # (3,) for dim=3, grade=1

# Bivector has geo=(d, d)
b = Vector(data, grade=2, metric=g)
b.geo   # (3, 3) for dim=3
```

### VectorSpec

Specifications for k-vector structure, used when defining Operators:

```python
from morphis.algebra import VectorSpec

# Scalar with batch of 10
VectorSpec(grade=0, lot=(10,), dim=3)

# Single bivector
VectorSpec(grade=2, lot=(), dim=3)

# Batch of vectors: shape (M, N, 3)
VectorSpec(grade=1, lot=(M, N), dim=3)
```

---

## Operator Overloading

Morphis uses Python's operator overloading to provide concise mathematical syntax.

### Algebraic Products

| Syntax   | Operation         | Description                                       |
| -------- | ----------------- | ------------------------------------------------- |
| `u ^ v`  | Wedge product     | Exterior product, creates higher-grade element    |
| `u * v`  | Geometric product | Full Clifford product (for Vectors → MultiVector) |
| `u << v` | Left contraction  | Interior product $u \, \lrcorner \, v$            |
| `u >> v` | Right contraction | Interior product $u \, \llcorner \, v$            |

```python
from morphis.elements import basis_vectors, euclidean_metric

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Wedge product: vector ^ vector → bivector
b = e1 ^ e2
b.grade  # 2

# Geometric product: vector * vector → multivector
M = e1 * e2
M.grades  # (0, 2) - scalar and bivector parts

# Left contraction: vector << bivector → vector
v = e1 << (e1 ^ e2)  # contracts e1 out, leaves e2
```

### Unary Operations

| Syntax | Operation | Description |
|--------|-----------|-------------|
| `~u` | Reverse | Reverses order of wedge factors |
| `-u` | Negation | Additive inverse |
| `u ** n` | Power | Integer powers (n = -1 gives inverse) |

```python
# Reverse: flips sign based on grade
b = e1 ^ e2
~b  # equals -b for bivectors

# Inverse via power
R = exp_vector(0.5 * (e1 ^ e2))  # rotor
R_inv = R ** (-1)  # multiplicative inverse

# Or explicitly
R.inv()  # same as R ** (-1)
```

### Scalar Operations

| Syntax | Operation | Description |
|--------|-----------|-------------|
| `3 * u` | Scalar multiply | Scale by constant |
| `u / 2` | Scalar divide | Divide by constant |
| `u + v` | Addition | Add same-grade elements |
| `u - v` | Subtraction | Subtract same-grade elements |

```python
# Scaling
v = 3 * e1 + 2 * e2
w = v / 5

# Addition requires matching grade
u = e1 + e2  # OK: both grade-1
# e1 + (e1 ^ e2)  # Error: grade mismatch
```

---

## Method Shortcuts

Many operations have three forms: long, short, and symbol.

| Long Form | Short Form | Symbol | Description |
|-----------|------------|--------|-------------|
| `reverse()` | `rev()` | `~x` | Reverse blade order |
| `inverse()` | `inv()` | `x**(-1)` | Multiplicative inverse |
| `conjugate()` | `conj()` | — | Complex conjugation |
| `unit()` | — | — | Unit normalization |
| `hodge()` | — | — | Hodge dual |

### Vector Methods

```python
v = Vector([3, 4, 0], grade=1, metric=g)

# Norms and unit vectors
v.form()       # quadratic form (v · v) = 25
v.norm()       # scalar magnitude = 5.0
v.unit()       # unit vector [0.6, 0.8, 0]

# Reverse and inverse
b = e1 ^ e2
b.rev()        # same as ~b
b.inv()        # same as b ** (-1)

# Hodge dual
b.hodge()      # grade-2 → grade-1 in 3D
```

### Vector Reductions

```python
# Batch of 100 vectors
v_batch = Vector(data, grade=1, metric=g, lot=(100,))

# Reduce over lot dimensions
v_batch.sum()           # sum all → lot=()
v_batch.mean()          # average → lot=()
v_batch.sum(axis=0)     # sum along first lot axis
```

### Operator Methods

| Long Form | Short Form | Symbol | Description |
|-----------|------------|--------|-------------|
| `adjoint()` | `adj()` | `.H` | Conjugate transpose |
| `transpose()` | `trans()` | `.T` | Transpose |
| `pseudoinverse()` | `pinv()` | — | Moore-Penrose inverse |

```python
L = Operator(...)

# Adjoint (conjugate transpose)
L.H          # property access
L.adj()      # method call
L.adjoint()  # verbose form

# Transpose
L.T          # property access

# Pseudoinverse
L.pinv()     # Moore-Penrose inverse

# SVD decomposition
U, S, Vt = L.svd()

# Least squares solve
x = L.solve(y, method="lstsq", alpha=1e-4)  # with regularization
x = L.solve(y, method="pinv")               # via pseudoinverse
```

---

## Tensor Contraction

Morphis provides two APIs for tensor contraction:

### Bracket Syntax (Preferred)

Use string indexing to create an `IndexedTensor`, then multiply:

```python
# Syntax: tensor["indices"] * tensor["indices"]
result = G["mnab"] * q["n"]  # contracts on matching index 'n'
```

```python
from morphis.elements import Vector, euclidean_metric

g = euclidean_metric(3)
u = Vector([1, 2, 3], grade=1, metric=g)
v = Vector([4, 5, 6], grade=1, metric=g)

# Dot product: matching indices are contracted
s = u["a"] * v["a"]  # s.data = 32

# Outer product: different indices, no contraction
outer = u["a"] * v["b"]  # shape (3, 3)

# Matrix-vector product
M = Vector(matrix_data, grade=2, metric=g)  # shape (3, 3)
w = M["ab"] * v["b"]  # contracts on 'b', result has index "a"

# Batch contraction
G = Vector(data, grade=2, metric=g, lot=(M, N))  # shape (M, N, 3, 3)
q = Vector(data, grade=0, metric=g, lot=(N,))    # shape (N,)
b = G["mnab"] * q["n"]  # contracts on 'n', result shape (M, 3, 3)
```

### Einsum-Style Function

The `contract` function works exactly like `numpy.einsum`:

```python
from morphis.algebra import contract

# Syntax: contract("signature", tensor1, tensor2, ...)
s = contract("a, a ->", u, v)           # dot product
outer = contract("a, b -> ab", u, v)    # outer product
w = contract("ab, b -> a", M, v)        # matrix-vector
b = contract("mnab, n -> mab", G, q)    # batch contraction

# Multi-way contraction
result = contract("mn, np, pm ->", A, B, C)  # all indices contracted
```

---

## Lot and Geo Accessors

Vectors provide accessors for clean indexing over lot and geometric dimensions:

### Lot Accessor (`.at`)

Index or slice over lot dimensions only, preserving geo:

```python
# v has lot=(10, 5), grade=1
v.at[0]       # first in first lot dim → lot=(5,)
v.at[3, 2]    # index both lot dims → lot=()
v.at[:, 0]    # slice first dim, index second → lot=(10,)
v.at[::2]     # every other in first dim → lot=(5, 5)
```

### Geometric Accessor (`.on`)

Index or slice over geo dimensions only, preserving lot:

```python
# b has grade=2, so geo=(d, d)
b.on[0, 1]    # component e_01 → grade=0
b.on[:, 0]    # first column → grade=1
b.on[0]       # first row → grade=1
```

Integer indices reduce grade; slices preserve it.

---

## Operator Class

The `Operator` class represents linear maps between k-vector spaces with full tensor structure.

### Creation

```python
from morphis.operations import Operator
from morphis.algebra import VectorSpec

# Transfer operator: N scalars → M bivectors
L = Operator(
    data=data,  # shape (M, N, d, d)
    input_spec=VectorSpec(grade=0, lot=(N,), dim=d),
    output_spec=VectorSpec(grade=2, lot=(M,), dim=d),
    metric=g,
)
```

### Application

```python
L = Operator(...)
v = Vector(...)

# All equivalent
y = L * v       # multiplication syntax
y = L(v)        # call syntax
y = L.apply(v)  # explicit method

# Composition
L_composed = L1 * L2  # L1(L2(x))
```

### Inversion

```python
# Least squares solve: find x such that L*x ≈ y
x = L.solve(y, method='lstsq')
x = L.solve(y, method='lstsq', alpha=1e-4)  # Tikhonov regularization
x = L.solve(y, method='pinv')               # via pseudoinverse

# Pseudoinverse operator
L_pinv = L.pinv()

# SVD decomposition
U, S, Vt = L.svd()
```

### Outermorphisms

When an Operator maps grade-1 to grade-1, it's an **outermorphism** and can apply to any grade:

```python
# Rotation operator (grade-1 → grade-1)
R = Operator(
    data=rotation_matrix,
    input_spec=VectorSpec(grade=1, lot=(), dim=3),
    output_spec=VectorSpec(grade=1, lot=(), dim=3),
    metric=g,
)

R.is_outermorphism  # True

# Apply to any grade
v_rotated = R * v           # grade-1
b_rotated = R * b           # grade-2 (uses exterior power)
M_rotated = R * M           # MultiVector (each grade transformed)
```

### Complex-Valued Operators

Operators support complex data for frequency-domain analysis:

```python
from numpy import exp, pi

# Complex transfer function
L_data = magnitude * exp(1j * phase)
L = Operator(data=L_data, ...)

# Complex phasor input
q_tilde = Vector(amplitude * exp(1j * theta), grade=0, metric=g)

# Forward and inverse
P_tilde = L * q_tilde
q_recovered = L.solve(P_tilde)
```

---

## Quick Reference

### Vector Creation

```python
from morphis.elements import Vector, basis_vectors, euclidean_metric

g = euclidean_metric(3)

# From data
v = Vector([1, 2, 3], grade=1, metric=g)

# Batched vectors
v_batch = Vector(data, grade=1, metric=g, lot=(100,))

# Basis vectors
e1, e2, e3 = basis_vectors(g)

# Higher grades via wedge
b = e1 ^ e2          # bivector
t = e1 ^ e2 ^ e3     # trivector (pseudoscalar in 3D)
```

### Common Operations

```python
from morphis.operations import (
    wedge, geometric, reverse, inverse,
    interior_left, interior_right,
    hodge_dual, form, norm, unit,
    exp_vector, log_versor, slerp,
)

# Products
wedge(u, v)           # same as u ^ v
geometric(u, v)       # same as u * v
interior_left(u, v)   # same as u << v

# Norms
form(b)               # quadratic form (b · b)
norm(b)               # scalar magnitude
unit(b)               # unit element

# Exponential/log
R = exp_vector(B)     # bivector → rotor
B = log_versor(R)     # rotor → bivector

# Interpolation
R_mid = slerp(R0, R1, 0.5)  # halfway rotation
```

### Frame Operations

```python
from morphis.elements import Frame

# Create a frame from vectors
F = Frame(e1, e2, e3)

# Transform entire frame
F_rotated = R * F * ~R

# Get individual vectors
v1 = F[0]

# Normalize all vectors
F_unit = F.unit()
```

### Transforms

```python
from morphis.transforms import rotor, rotate

# Create rotor for rotation
b = (e1 ^ e2).unit()
R = rotor(b, angle)

# Apply rotation
v_rotated = rotate(v, R)  # or: R * v * ~R
```
