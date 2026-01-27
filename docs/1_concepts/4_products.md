# Products in Geometric Algebra

Geometric algebra provides several products that encode different geometric relationships. The **wedge product** builds higher-grade elements, the **interior product** contracts grades, and the **geometric product** unifies them all.

## The Wedge Product

The **wedge product** (exterior product) constructs higher-grade Vectors by combining lower-grade ones.

### Definition

For grade-$j$ k-vector $u$ and grade-$k$ k-vector $v$, the wedge product $u \wedge v$ is a grade-$(j+k)$ k-vector:

$$
(u \wedge v)^{m_1 \ldots m_{j+k}} = \frac{1}{j! \, k!} \, u^{[m_1 \ldots m_j} v^{m_{j+1} \ldots m_{j+k}]}
$$

where brackets denote antisymmetrization.

### Properties

**Anticommutativity:**

$$
u \wedge v = (-1)^{jk} \, v \wedge u
$$

For grade-1 vectors:

$$\mathbf{u} \wedge \mathbf{v} = -\mathbf{v} \wedge \mathbf{u}$$

**Nilpotency:**

$$\mathbf{v} \wedge \mathbf{v} = 0$$

Linear dependence:

$$\mathbf{u} = \alpha \mathbf{v} \implies \mathbf{u} \wedge \mathbf{v} = 0$$

**Associativity:**

$$(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})$$

### Usage in Morphis

```python
from morphis.elements import basis_vectors, euclidean_metric
from morphis.operations import wedge

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Operator syntax
b = e1 ^ e2          # Bivector
t = e1 ^ e2 ^ e3     # Trivector (pseudoscalar in 3D)

# Function syntax
b = wedge(e1, e2)
```

### Geometric Interpretation

The wedge product $\mathbf{u} \wedge \mathbf{v}$ represents the **oriented parallelogram** spanned by $\mathbf{u}$ and $\mathbf{v}$:
- Magnitude: area of the parallelogram
- Orientation: determined by the order of vectors

For $k$ vectors, $\mathbf{v}_1 \wedge \cdots \wedge \mathbf{v}_k$ represents the oriented $k$-volume of the parallelepiped they span.

## The Interior Product

The **interior product** (contraction) reduces grade by contracting indices using the metric.

### Left Contraction

For grade-$j$ k-vector $u$ and grade-$k$ k-vector $v$ with $j \leq k$:

$$
(u \lrcorner v)^{n_1 \ldots n_{k-j}} = u^{m_1 \ldots m_j} v_{m_1 \ldots m_j}^{\ \ \ \ \ \ \ \ n_1 \ldots n_{k-j}}
$$

Result grade: $k - j$

When $j > k$: $u \lrcorner v = 0$

### Right Contraction

$$
(u \llcorner v)^{m_1 \ldots m_{j-k}} = u_{n_1 \ldots n_k}^{m_1 \ldots m_{j-k}} v^{n_1 \ldots n_k}
$$

Result grade: $j - k$

When $k > j$: $u \llcorner v = 0$

### Usage in Morphis

```python
from morphis.elements import basis_vectors, euclidean_metric
from morphis.operations import interior_left, interior_right

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)
b = e1 ^ e2

# Left contraction: << operator
result = e1 << b  # Grade 2-1 = 1 (vector)

# Right contraction: >> operator
result = b >> e1  # Grade 2-1 = 1 (vector)

# Function syntax
interior_left(e1, b)
interior_right(b, e1)
```

### Geometric Interpretation

The interior product $v \lrcorner b$ gives the component of $b$ "perpendicular" to $v$, with one dimension removed. It's the algebraic counterpart of orthogonal projection.

## The Dot Product

For grade-1 vectors, the **dot product** extracts the scalar (symmetric) part:

$$\mathbf{u} \cdot \mathbf{v} = g_{ab} u^a v^b$$

This equals the full interior product when both operands are grade-1.

```python
from morphis.operations import dot

scalar = dot(u, v)  # Returns a scalar (grade-0 Vector)
```

## The Geometric Product

The **geometric product** is the fundamental operation of Clifford algebra, combining inner and outer products.

### For Vectors (Grade-1)

$$\mathbf{a} \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \wedge \mathbf{b}$$

The symmetric part gives the dot product (scalar), the antisymmetric part gives the wedge product (bivector).

### General Form

For general multivectors, the geometric product distributes over grades:

$$
MN = \sum_{r,s} \sum_{t = |r-s|}^{r+s} \langle M_r N_s \rangle_t
$$

where $M_r = \langle M \rangle_r$ and the sum over $t$ has step 2 (parity preservation).

### Usage in Morphis

```python
from morphis.elements import basis_vectors, euclidean_metric
from morphis.operations import geometric

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Operator syntax: *
M = e1 * e2  # MultiVector with grades {0, 2}

# Orthogonal vectors: pure bivector
(e1 * e2).grades  # [2]

# Parallel vectors: pure scalar
(e1 * e1).grades  # [0]

# Function syntax
M = geometric(e1, e2)
```

### Properties

**Associativity:**

$$
(MN)P = M(NP)
$$

**Distributivity:**

$$
M(N + P) = MN + MP
$$

**Not commutative (in general):**

$$
MN \neq NM
$$

### Relationship to Other Products

The wedge and interior products can be extracted from the geometric product. For k-vectors $u$ and $v$ of grades $j$ and $k$:

$$
u \wedge v = \langle uv \rangle_{j + k}
$$

$$
u \cdot v = \langle uv \rangle_{|j - k|}
$$

## The Antiwedge Product (Meet)

The **antiwedge** (regressive product, meet) finds the intersection of subspaces. For k-vectors $u$ and $v$:

$$
u \vee v = \overline{\left(\overline{u} \wedge \overline{v}\right)}
$$

where $\overline{\phantom{x}}$ denotes the complement.

```python
from morphis.operations import antiwedge, meet

# Equivalent operations
intersection = antiwedge(u, v)
intersection = meet(u, v)
```

## The Commutator and Anticommutator

The **commutator product**:

$$
[M, N] = \frac{1}{2}(MN - NM)
$$

The **anticommutator product**:

$$
\{M, N\} = \frac{1}{2}(MN + NM)
$$

```python
from morphis.operations import commutator, anticommutator

C = commutator(M, N)
AC = anticommutator(M, N)
```

The commutator of bivectors generates the Lie algebra structure of rotations.

## Scalar Product

The **scalar product** extracts only the grade-0 part of the geometric product:

$$
M * N = \langle MN \rangle_0
$$

```python
from morphis.operations import scalar_product

s = scalar_product(M, N)  # Grade-0 only
```

## Summary Table

| Product | Symbol | Result Grade | Meaning |
|---------|--------|--------------|---------|
| Wedge | $\wedge$ / `^` | $j + k$ | Span, oriented volume |
| Interior (left) | $\lrcorner$ / `<<` | $k - j$ | Contraction |
| Interior (right) | $\llcorner$ / `>>` | $j - k$ | Contraction |
| Geometric | juxtaposition / `*` | Mixed | Full algebraic product |
| Dot | $\cdot$ | 0 | Scalar inner product |
| Antiwedge | $\vee$ | $j + k - d$ | Intersection |
| Scalar | $*$ | 0 | Grade-0 extraction |
