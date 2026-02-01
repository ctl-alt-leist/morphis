# Mathematical Foundations

This document establishes the mathematical structures underlying geometric algebra: vector spaces, tensor products, and exterior algebra.

## Vector Spaces

We begin with a finite-dimensional real vector space $V$ of dimension $d$. The **dual space** $V^*$ consists of all linear functionals $\omega: V \to \mathbb{R}$. The natural pairing between $V$ and $V^*$ is given by evaluation:

$$\langle \omega, v \rangle = \omega(v)$$

Given a basis $\{\mathbf{e}_m\}$ for $V$, there exists a unique **dual basis** $\{\mathbf{e}^m\}$ for $V^*$ satisfying:

$$\mathbf{e}^m(\mathbf{e}_n) = \delta^m_n$$

In morphis, the dimension is determined by the `Metric`:

```python
from morphis.elements import euclidean_metric

g = euclidean_metric(3)  # 3-dimensional Euclidean space
g.dim  # 3
```

## Naming Conventions

Morphis uses consistent variable naming throughout documentation and code:

| Type                | Case      | Default Names | Notes                                    |
| ------------------- | --------- | ------------- | ---------------------------------------- |
| Vectors (any grade) | Lower | `u, v, w`     | Grade-1, 2, 3, etc. all use Lower    |
| Blades              | Lower | `b`           | When emphasizing factorizability         |
| Multivectors        | Upper | `M, N, R, S`  | Mixed-grade elements                     |
| Rotors              | Upper | `R`           | Even multivector with $R\tilde{R} = 1$   |
| Metrics             | Lower | `g, h, eta`   | `g` Euclidean, `h` PGA, `eta` Lorentzian |

This convention emphasizes that a bivector is still a vector (in $\bigwedge^2 V$), not a multivector.

## Basis Indexing Conventions

Morphis uses **1-based indexing** for Euclidean basis vectors, matching standard mathematical notation:

$$\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3 \quad \leftrightarrow \quad x, y, z$$

The index 0 is reserved for special directions that extend Euclidean space:

| Algebra | $\mathbf{e}_0$ meaning | Euclidean basis |
|---------|------------------------|-----------------|
| **VGA** (Euclidean) | *not used* | $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ |
| **PGA** (Projective) | ideal direction ($\mathbf{e}_0^2 = 0$) | $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ |
| **Lorentzian** | time direction ($\mathbf{e}_0^2 = -1$) | $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ |

In code, `basis_vectors()` returns a tuple indexed from 0, but we name them according to the algebra:

```python
from morphis.elements import euclidean_metric, pga_metric, basis_vectors

# Euclidean 3D: name them e1, e2, e3
g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# PGA 3D: e0 is the ideal direction
h = pga_metric(3)
e0, e1, e2, e3 = basis_vectors(h)
```

## Tensors

A **$(p,q)$-tensor** lives in the space $V^{\otimes p} \otimes (V^*)^{\otimes q}$, with $p$ contravariant indices (upstairs) and $q$ covariant indices (downstairs). In components relative to a basis:

$$T = T^{m_1 \ldots m_p}_{n_1 \ldots n_q} \, \mathbf{e}_{m_1} \otimes \cdots \otimes \mathbf{e}_{m_p} \otimes \mathbf{e}^{n_1} \otimes \cdots \otimes \mathbf{e}^{n_q}$$

The `Tensor` class in morphis stores tensors with this structure:

```python
from morphis.elements import Tensor, euclidean_metric

g = euclidean_metric(3)
T = Tensor(data, contravariant=2, covariant=0, metric=g)
```

Under a change of basis $\mathbf{e}_m' = R^n_m \mathbf{e}_n$, tensor components transform to preserve the tensor itself. This transformation law is a **consequence** of the tensor being a basis-independent geometric object, not a definition.

## The Exterior Algebra

The **exterior algebra** $\bigwedge V$ consists of completely antisymmetric tensors:

$$\bigwedge V = \bigoplus_{k=0}^{d} \bigwedge^k V$$

The $k$-th exterior power $\bigwedge^k V$ has dimension $\binom{d}{k}$. Elements of $\bigwedge^k V$ are called **$k$-vectors** (or homogeneous multivectors of grade $k$).

The **wedge product** of vectors:

$$\mathbf{a} \wedge \mathbf{b} = \mathbf{a} \otimes \mathbf{b} - \mathbf{b} \otimes \mathbf{a}$$

In components:

$$(\mathbf{a} \wedge \mathbf{b})^{mn} = a^m b^n - a^n b^m$$

The wedge product is:
- **Anticommutative**: $\mathbf{a} \wedge \mathbf{b} = -\mathbf{b} \wedge \mathbf{a}$
- **Associative**: $(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})$
- **Nilpotent**: $\mathbf{v} \wedge \mathbf{v} = 0$

In morphis:

```python
from morphis.elements import basis_vectors, euclidean_metric

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Wedge product creates a bivector
b = e1 ^ e2  # grade-2 vector
b.grade  # 2
```

## Storage Convention

Morphis stores k-vectors using **full antisymmetric tensor storage**. A grade-$k$ element in $d$-dimensional space has shape $(*\text{lot}, d, d, \ldots, d)$ with $k$ copies of $d$.

This redundant storage enables:
- Direct einsum operations without index bookkeeping
- Uniform batch dimension handling via `...`
- Simple grade-agnostic algorithms

The antisymmetry constraint $b^{\ldots m \ldots n \ldots} = -b^{\ldots n \ldots m \ldots}$ is maintained by all operations.

### Antisymmetry on Components

The basis k-vectors $\mathbf{e}_{mn\ldots}$ are conventionally written as antisymmetric:

$$
\mathbf{e}_{mn} = \mathbf{e}_m \wedge \mathbf{e}_n = -\mathbf{e}_{nm}
$$

However, in computation we need a concrete representation. Morphis uses an **ordered basis** with antisymmetry carried on the components. We write:

$$
\mathbf{e}^{<}_{mn} \quad \text{(ordered basis, indices satisfy } m < n \text{)}
$$

Any k-vector can then be expressed in two equivalent ways:

$$
b = b^{mn} \mathbf{e}_{mn}
  = \tilde{b}^{mn} \mathbf{e}^{<}_{mn}
$$

where the "symmetric" components $b^{mn}$ are used with the unordered (conceptual) basis, and the antisymmetric components $\tilde{b}^{mn}$ incorporate the full alternating structure:

$$
\tilde{b}^{mn} = b^{mn} \, \varepsilon^{mn}
$$

The $k$-index antisymmetric symbol $\varepsilon^{m_1 \ldots m_k}$ automatically handles sign changes from index ordering. For a bivector in 2D:

$$
b = \frac{1}{2}(u^1 v^2 - u^2 v^1) \, \mathbf{e}_{12}
$$

The factor $\frac{1}{2}$ ensures proper normalization, while the antisymmetric combination arises from the Levi-Civita structure.

This convention allows efficient computation: we sum over ordered index combinations and let the antisymmetric symbol handle the signs, rather than explicitly storing and manipulating signed basis elements.

## Lot Dimensions

All morphis elements support leading **lot dimensions** (also called batch dimensions). This is fundamental to efficient computation and distinguishes geometric dimensions from batch dimensions.

### The Storage Convention

Every element has shape `(*lot, *geo)`:
- **Lot dimensions** come first (leftmost): arbitrary shape for batching
- **Geometric dimensions** come last (rightmost): determined by grade and dimension

```python
from numpy import random
from morphis.elements import Vector, euclidean_metric

g = euclidean_metric(3)

# Single vector: lot=(), geo=(3,)
v = Vector([1, 0, 0], grade=1, metric=g)
v.data.shape  # (3,)
v.lot  # ()

# 10 vectors: lot=(10,), geo=(3,)
u = Vector(random.randn(10, 3), grade=1, metric=g, lot=(10,))
u.data.shape  # (10, 3)
u.lot  # (10,)

# 5x10 grid of bivectors: lot=(5, 10), geo=(3, 3)
b = Vector(random.randn(5, 10, 3, 3), grade=2, metric=g, lot=(5, 10))
b.data.shape  # (5, 10, 3, 3)
b.lot  # (5, 10)
b.grade       # 2
```

### Geometric Dimensions by Grade

The number of geometric dimensions equals the grade:

| Grade | Geometric Shape | Example |
|-------|----------------|---------|
| 0 | `()` | Scalar has no geometric indices |
| 1 | `(d,)` | Vector has 1 index |
| 2 | `(d, d)` | Bivector has 2 indices |
| $k$ | `(d,) * k` | $k$-vector has $k$ indices |

### Broadcasting

Lot dimensions broadcast according to NumPy rules:

```python
# Shape (10, 3) broadcasts with shape (3,)
u_batch = Vector(random.randn(10, 3), grade=1, metric=g, lot=(10,))
v_single = Vector([1, 0, 0], grade=1, metric=g)

# Wedge product: (10, 3) ^ (3,) -> (10, 3, 3)
b = u_batch ^ v_single
b.lot  # (10,)

# Two batches: (5, 1, 3) ^ (1, 10, 3) -> (5, 10, 3, 3)
```

This enables batch operations without explicit loops, providing orders of magnitude speedup for parallel computations.

## Einsum Patterns

Morphis uses NumPy's `einsum` for efficient tensor contractions. The key insight: the ellipsis `...` absorbs all lot dimensions automatically.

### The Core Pattern

All operations follow the pattern:
```python
einsum("fixed_indices, ...geometric1, ...geometric2 -> ...result", ...)
```

The `...` matches any number of leading lot dimensions, and the explicit indices handle the geometric contraction.

### Example: Dot Product

For two grade-1 vectors, the dot product contracts via the metric:

```python
# Mathematical: g_{ab} u^a v^b
# Einsum: metric ab, vector a, vector b -> scalar
einsum("ab, ...a, ...b -> ...", g, u.data, v.data)
```

This single line handles:
- Single vectors: `(d,)` + `(d,)` → `()`
- Batched: `(N, d)` + `(d,)` → `(N,)`
- Multi-batched: `(M, N, d)` + `(K, d)` → `(M, N, K)`

### Example: Wedge Product

For two grade-1 vectors forming a bivector:

```python
# Mathematical: a^m b^n - a^n b^m (antisymmetrize)
# Einsum: vector a, vector b -> tensor ab, then antisymmetrize
temp = einsum("...a, ...b -> ...ab", u.data, v.data)
result = temp - einsum("...ab -> ...ba", temp)  # antisymmetrize
```

### Example: Interior Product

Left contraction of a vector into a bivector:

```python
# Mathematical: g_{am} u^m b^{an}  (contract first index)
# Einsum: metric am, vector m, bivector an -> vector n
einsum("am, ...m, ...an -> ...n", g, u.data, b.data)
```

### Grade-Dependent Signatures

For operations on arbitrary grades, the einsum signature depends on the grade. Morphis generates these dynamically and caches them:

```python
# Grade-2 norm: g_{ac} g_{bd} b^{ab} b^{cd}
einsum("ac, bd, ...ab, ...cd -> ...", g, g, b.data, b.data) / 2

# Grade-3 norm: g_{ad} g_{be} g_{cf} t^{abc} t^{def}
einsum("ad, be, cf, ...abc, ...def -> ...", g, g, g, t.data, t.data) / 6
```

The signature generation is in `morphis.operations.structure`.

### Why This Approach

The einsum-based approach provides:

1. **Manifest generality**: The code is the mathematical expression
2. **Automatic broadcasting**: Lot dimensions handled uniformly
3. **Efficiency**: NumPy optimizes contraction order
4. **Clarity**: Mathematical intent is visible in the signature

The cost is that grade-varying operations require signature generation. Morphis mitigates this by caching signatures and keeping generation logic isolated.

## Dimension Counting

The exterior algebra has total dimension $2^d$. Grade by grade:

| Grade $k$ | Dimension $\binom{d}{k}$ | Name |
|-----------|------------------------|------|
| 0 | 1 | Scalar |
| 1 | $d$ | Vector |
| 2 | $\binom{d}{2}$ | Bivector |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $d$ | 1 | Pseudoscalar |

For $d = 3$: $1 + 3 + 3 + 1 = 8 = 2^3$ total basis elements.

```python
from morphis.elements import geometric_basis, euclidean_metric

g = euclidean_metric(3)
basis = geometric_basis(g)

len(basis[0])  # 1 (scalar)
len(basis[1])  # 3 (vectors)
len(basis[2])  # 3 (bivectors)
len(basis[3])  # 1 (pseudoscalar)
```
