# Elements: Mathematical Properties

Mathematical foundations for blades, multivectors, metrics, and their algebraic structure.

## Metric Tensor

The metric tensor $g_{ab}$ defines the inner product structure of the vector space.

### PGA Metric

For projective geometric algebra embedding $d$-dimensional Euclidean space:

$$
g = \text{diag}(0, 1, 1, \ldots, 1)
$$

Properties:
- $g_{00} = 0$ (degenerate ideal direction)
- $g_{mm} = 1$ for $m \geq 1$ (Euclidean subspace)
- $\dim(g) = d + 1$

### Euclidean Metric

The standard Euclidean metric in $d$ dimensions:

$$
g = \mathbb{I}_d = \text{diag}(1, 1, \ldots, 1)
$$

## Blade Construction

A $k$-blade in $d$-dimensional space has components $B^{m_1 \ldots m_k}$ satisfying antisymmetry:

$$
B^{\ldots m \ldots n \ldots} = -B^{\ldots n \ldots m \ldots}
$$

### Grade-Specific Structure

**Vectors (grade 1):**

$$
\mathbf{v} = v^m \mathbf{e}_m
$$

Shape: $(d,)$

**Bivectors (grade 2):**

$$
\mathbf{B} = B^{mn} \mathbf{e}_{mn}
$$

Shape: $(d, d)$ with antisymmetric components.

**Trivectors (grade 3):**

$$
\mathbf{T} = T^{mnp} \mathbf{e}_{mnp}
$$

Shape: $(d, d, d)$ with full antisymmetry.

### Collection Dimensions

Batch processing uses leading collection dimensions:

$$
\text{shape} = (*\text{batch}, *\text{geometric})
$$

where:
- $\text{cdim}$ = number of leading batch dimensions
- $\text{ndim} = \text{cdim} + \text{grade}$

For batch shape $(N,)$ with vectors in $\mathbb{R}^d$: storage is $(N, d)$ with $\text{cdim} = 1$.

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

$$
(\mathbf{A} + \mathbf{B})^{m_1 \ldots m_k} = A^{m_1 \ldots m_k} + B^{m_1 \ldots m_k}
$$

Preserves grade and dimension:

$$
\text{grade}(\mathbf{A} + \mathbf{B}) = \text{grade}(\mathbf{A}) = \text{grade}(\mathbf{B})
$$

### Scalar Multiplication

For scalar $\alpha$ and blade $\mathbf{B}$:

$$
(\alpha \mathbf{B})^{m_1 \ldots m_k} = \alpha B^{m_1 \ldots m_k}
$$

Commutative: $\alpha \mathbf{B} = \mathbf{B} \alpha$

### Negation

$$
(-\mathbf{B})^{m_1 \ldots m_k} = -B^{m_1 \ldots m_k}
$$

Satisfies $\mathbf{B} + (-\mathbf{B}) = 0$.

### Type Constraints

Operations require matching grade and dimension:

$$
\text{grade}(\mathbf{A}) \neq \text{grade}(\mathbf{B}) \implies \mathbf{A} + \mathbf{B} \text{ undefined}
$$

## MultiVector Structure

A multivector combines blades of different grades:

$$
\mathbf{M} = \sum_{k=0}^{d} \mathbf{M}_{\langle k \rangle}
$$

### Grade Selection

The grade projection:

$$
\langle \mathbf{M} \rangle_k = \mathbf{M}_{\langle k \rangle}
$$

extracts the grade-$k$ component.

### Multivector Addition

$$
(\mathbf{M} + \mathbf{N})_{\langle k \rangle} = \mathbf{M}_{\langle k \rangle} + \mathbf{N}_{\langle k \rangle}
$$

Each grade adds independently.

### Construction from Blades

Given blades $\mathbf{B}_1, \ldots, \mathbf{B}_n$:

$$
\mathbf{M}_{\langle k \rangle} = \sum_{i: \text{grade}(\mathbf{B}_i) = k} \mathbf{B}_i
$$

## Utility Functions

### Permutation Sign

$$
\text{sgn}(\sigma) = \begin{cases}
+1 & \text{if } \sigma \text{ is even} \\
-1 & \text{if } \sigma \text{ is odd}
\end{cases}
$$

Examples:
- Identity: $\text{sgn}((0, 1, 2)) = +1$
- Single swap: $\text{sgn}((1, 0, 2)) = -1$
- Forward cycle: $\text{sgn}((1, 2, 0)) = +1$

### Antisymmetrization

For rank-$k$ tensor $T$:

$$
T^{[m_1 \ldots m_k]} = \sum_{\sigma \in S_k} \text{sgn}(\sigma) \, T^{m_{\sigma(1)} \ldots m_{\sigma(k)}}
$$

Properties:
- Index swap reverses sign
- Repeated indices vanish: $T^{[m m n]} = 0$

### Levi-Civita Symbol

$$
\varepsilon^{m_1 \ldots m_d} = \begin{cases}
+1 & \text{even permutation of } (0, \ldots, d-1) \\
-1 & \text{odd permutation} \\
0 & \text{repeated indices}
\end{cases}
$$

Self-contraction:

$$
\varepsilon^{abc} \varepsilon_{abc} = d!
$$

### Generalized Kronecker Delta

$$
\delta^{m_1 \ldots m_k}_{n_1 \ldots n_k} = \frac{1}{k!} \sum_{\sigma \in S_k} \text{sgn}(\sigma) \, \delta^{m_1}_{n_{\sigma(1)}} \cdots \delta^{m_k}_{n_{\sigma(k)}}
$$

Antisymmetric in both upper and lower indices.
