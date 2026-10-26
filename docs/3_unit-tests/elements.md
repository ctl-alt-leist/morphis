# Elements: Mathematical Properties

Mathematical foundations for blades, multivectors, metrics, and their algebraic structure.

## Metric Tensor

The metric tensor $g_{ab}$ defines the inner product structure of the vector space.

### PGA Metric

For projective geometric algebra embedding $d$-dimensional Euclidean space:

```math
g = \text{diag}(0, 1, 1, \ldots, 1)
```

Properties:
- $g_{00} = 0$ (degenerate ideal direction)
- $g_{mm} = 1$ for $m \geq 1$ (Euclidean subspace)
- $\dim(g) = d + 1$

### Euclidean Metric

The standard Euclidean metric in $d$ dimensions:

```math
g = \mathbb{I}_d = \text{diag}(1, 1, \ldots, 1)
```

### Block Layout

A space is declared by its blocks. Storage order is (time, space, projective), or (projective, space) for a purely spatial space with one projective direction, and labels follow position with $x$ always 1:

| Declaration | Diagonal | Labels |
|-------------|----------|--------|
| `metric(space=3)` | $(1, 1, 1)$ | $1, 2, 3$ |
| `metric(space=3, projective=1)` | $(0, 1, 1, 1)$ | $0, 1, 2, 3$ |
| `metric(time=1, space=3)` | $(1, -1, -1, -1)$ | $0, 1, 2, 3$ |
| `metric(time=1, space=3, projective=1)` | $(1, -1, -1, -1, 0)$ | $0, 1, 2, 3, 4$ |
| `metric(modes=2)` | $(1, 1, 1, 1)$ | $1, \dot 1, 2, \dot 2$ |

Properties:
- `signature` is the inertia $(p, q, r)$ of the tensor
- `convention="mostly plus"` flips every sign of a space with time
- Metrics are equal when layout and tensor agree: $\mathbb{R}^{2,2} \neq \mathbb{R}^{1,3}$ and $\mathbb{R}^4 \neq \mathbb{C}^2$

## Blade Construction

A $k$-blade in $d$-dimensional space has components $B^{m_1 \ldots m_k}$ satisfying antisymmetry:

```math
B^{\ldots m \ldots n \ldots} = -B^{\ldots n \ldots m \ldots}
```

### Grade-Specific Structure

**Vectors (grade 1):**

```math
\mathbf{v} = v^m \mathbf{e}_m
```

Shape: $(d,)$

**Bivectors (grade 2):**

```math
\mathbf{B} = B^{mn} \mathbf{e}_{mn}
```

Shape: $(d, d)$ with antisymmetric components.

**Trivectors (grade 3):**

```math
\mathbf{T} = T^{mnp} \mathbf{e}_{mnp}
```

Shape: $(d, d, d)$ with full antisymmetry.

### Lot Dimensions

Batch processing uses leading lot dimensions:

```math
\text{shape} = (*\text{lot}, *\text{geo})
```

where:
- lot dimensions = leading batch dimensions
- geo dimensions = trailing geometric dimensions (determined by grade)
- $\text{ndim} = \text{len(lot)} + \text{grade}$

For lot shape $(N,)$ with vectors in $\mathbb{R}^d$: storage is $(N, d)$.

### Constraints

Valid blades satisfy:
- $\text{grade} \geq 0$
- $\text{cdim} \geq 0$
- $\text{ndim} = \text{cdim} + \text{grade}$
- $\text{shape}[-k] = d$ for all $k \in [1, \text{grade}]$

## Blade Arithmetic

Blades of equal grade form a vector space.

### Addition

For blades $\mathbf{A}$, $\mathbf{B}$ of grade $k$:

```math
(\mathbf{A} + \mathbf{B})^{m_1 \ldots m_k} = A^{m_1 \ldots m_k} + B^{m_1 \ldots m_k}
```

Preserves grade and dimension:

```math
\text{grade}(\mathbf{A} + \mathbf{B}) = \text{grade}(\mathbf{A}) = \text{grade}(\mathbf{B})
```

### Scalar Multiplication

For scalar $α$ and blade $\mathbf{B}$:

```math
(α \mathbf{B})^{m_1 \ldots m_k} = α B^{m_1 \ldots m_k}
```

Commutative: $α \mathbf{B} = \mathbf{B} α$

### Negation

```math
(-\mathbf{B})^{m_1 \ldots m_k} = -B^{m_1 \ldots m_k}
```

Satisfies $\mathbf{B} + (-\mathbf{B}) = 0$.

### Type Constraints

Operations require matching grade and dimension:

```math
\text{grade}(\mathbf{A}) \neq \text{grade}(\mathbf{B}) \implies \mathbf{A} + \mathbf{B} \text{ undefined}
```

## MultiVector Structure

A multivector combines blades of different grades:

```math
\mathbf{M} = \sum_{k=0}^{d} \mathbf{M}_{\langle k \rangle}
```

### Grade Selection

The grade projection:

```math
\langle \mathbf{M} \rangle_k = \mathbf{M}_{\langle k \rangle}
```

extracts the grade-$k$ component.

### Multivector Addition

```math
(\mathbf{M} + \mathbf{N})_{\langle k \rangle} = \mathbf{M}_{\langle k \rangle} + \mathbf{N}_{\langle k \rangle}
```

Each grade adds independently.

### Construction from Blades

Given blades $\mathbf{B}_1, \ldots, \mathbf{B}_n$:

```math
\mathbf{M}_{\langle k \rangle} = \sum_{i: \text{grade}(\mathbf{B}_i) = k} \mathbf{B}_i
```

## Linear Operators

An Operator represents a structured linear map $L: V \to W$ between blade spaces.

### VectorSpec

A `VectorSpec` defines the structure of a vector space:

```math
\text{VectorSpec} = (\text{grade}, \text{lot}, \text{dim})
```

- $\text{grade}$: The grade of vectors (0=scalar, 1=vector, 2=bivector, etc.)
- $\text{lot}$: Shape of batch/lot dimensions
- $\text{dim}$: Dimension of the underlying vector space

### Operator Storage (Lot-First Layout)

Operator data is stored with lot-first layout:

```math
\text{shape} = (*\text{output\_lot}, *\text{input\_lot}, *\text{output\_geo}, *\text{input\_geo})
```

For example, an operator mapping scalar currents $(N,)$ to bivector fields $(M, d, d)$ has data shape $(M, N, d, d)$.

### Operator Properties

| Property | Description |
|----------|-------------|
| `input_lot` | Shape of input lot dimensions |
| `output_lot` | Shape of output lot dimensions |
| `input_shape` | Full expected shape for input vector data |
| `output_shape` | Full shape of output vector data |

### Operator Methods

**Forward Application:**
```math
y = L(x)
```

Via `L * x` or `L.apply(x)` or `L(x)`.

**Adjoint (conjugate transpose):**
```math
\langle Lx, y \rangle = \langle x, L^H y \rangle
```

Via `L.adjoint()`, `L.adj()`, or `L.H`.

**Transpose:**
```math
L^T
```

Via `L.transpose()`, `L.trans()`, or `L.T`.

**Pseudoinverse:**
```math
L^+ : L L^+ L = L, \quad L^+ L L^+ = L^+
```

Via `L.pseudoinverse()` or `L.pinv()`.

**SVD Decomposition:**
```math
L = U \cdot \text{diag}(S) \cdot V^T
```

Via `U, S, Vt = L.svd()`.

**Composition:**
```math
(L_1 \circ L_2)(x) = L_1(L_2(x))
```

Via `L1 * L2` or `L1.compose(L2)`.

## Utility Functions

### Permutation Sign

```math
\text{sgn}(σ) = \begin{cases}
+1 & \text{if } σ \text{ is even} \\
-1 & \text{if } σ \text{ is odd}
\end{cases}
```

Examples:
- Identity: $\text{sgn}((0, 1, 2)) = +1$
- Single swap: $\text{sgn}((1, 0, 2)) = -1$
- Forward cycle: $\text{sgn}((1, 2, 0)) = +1$

### Antisymmetrization

For rank-$k$ tensor $T$:

```math
T^{[m_1 \ldots m_k]} = \sum_{σ \in S_k} \text{sgn}(σ) T^{m_{σ(1)} \ldots m_{σ(k)}}
```

Properties:
- Index swap reverses sign
- Repeated indices vanish: $T^{[m m n]} = 0$

### Levi-Civita Symbol

```math
ε^{m_1 \ldots m_d} = \begin{cases}
+1 & \text{even permutation of } (0, \ldots, d-1) \\
-1 & \text{odd permutation} \\
0 & \text{repeated indices}
\end{cases}
```

Self-contraction:

```math
ε^{abc} ε_{abc} = d!
```

### Generalized Kronecker Delta

```math
δ^{m_1 \ldots m_k}_{n_1 \ldots n_k} = \frac{1}{k!} \sum_{σ \in S_k} \text{sgn}(σ) δ^{m_1}_{n_{σ(1)}} \cdots δ^{m_k}_{n_{σ(k)}}
```

Antisymmetric in both upper and lower indices.
