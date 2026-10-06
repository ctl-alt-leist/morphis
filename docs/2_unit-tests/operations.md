# Operations: Mathematical Properties

Mathematical foundations for wedge product, interior product, duality, norms, and projections.

## Wedge Product

The wedge (exterior) product constructs higher-grade blades.

### Definition

For grade-$j$ blade $\mathbf{A}$ and grade-$k$ blade $\mathbf{B}$:

```math
(\mathbf{A} \wedge \mathbf{B})^{m_1 \ldots m_{j + k}} = \frac{1}{j! k!} A^{m_1 \ldots m_j} B^{m_{j + 1} \ldots m_{j + k}} \times \text{antisymmetrize}
```

Result has grade $(j + k)$.

### Anticommutativity

For vectors:

```math
\mathbf{u} \wedge \mathbf{v} = -\mathbf{v} \wedge \mathbf{u}
```

General form:

```math
\mathbf{A} \wedge \mathbf{B} = (-1)^{jk} \mathbf{B} \wedge \mathbf{A}
```

### Nilpotency

```math
\mathbf{v} \wedge \mathbf{v} = 0
```

Linear dependence:

```math
\mathbf{u} = \alpha \mathbf{v} \implies \mathbf{u} \wedge \mathbf{v} = 0
```

### Associativity

```math
(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})
```

### Grade Calculation

```math
\text{grade}(\mathbf{A} \wedge \mathbf{B}) = \text{grade}(\mathbf{A}) + \text{grade}(\mathbf{B})
```

## Interior Product

The interior product (left contraction) contracts indices using the metric.

### Definition

For grade-$j$ blade $\mathbf{A}$ and grade-$k$ blade $\mathbf{B}$ with $j \leq k$:

```math
(\mathbf{A} \lrcorner \mathbf{B})^{n_1 \ldots n_{k - j}} = A^{m_1 \ldots m_j} B_{m_1 \ldots m_j}^{n_1 \ldots n_{k - j}}
```

Indices lowered via metric:

```math
B_{m_1 \ldots m_j}^{n_1 \ldots n_{k - j}} = g_{m_1 p_1} \cdots g_{m_j p_j} B^{p_1 \ldots p_j n_1 \ldots n_{k - j}}
```

### Grade Reduction

```math
\text{grade}(\mathbf{A} \lrcorner \mathbf{B}) = \text{grade}(\mathbf{B}) - \text{grade}(\mathbf{A})
```

### Special Cases

When $j > k$:

```math
\mathbf{A} \lrcorner \mathbf{B} = 0
```

Scalar contraction:

```math
s \lrcorner \mathbf{B} = s \mathbf{B}
```

### Orthogonality

For orthogonal basis vectors $\mathbf{e}_m \perp \mathbf{e}_n$ with $m \neq n$:

```math
\mathbf{e}_m \lrcorner (\mathbf{e}_n \wedge \mathbf{e}_p) = 0 \quad \text{when } m \notin \{n, p\}
```

## Complement Operations

Complement operations map between dual grades using the Levi-Civita symbol.

### Right Complement

For grade-$k$ blade in $d$ dimensions:

```math
\overline{\mathbf{B}}^{m_{k + 1} \ldots m_d} = B^{m_1 \ldots m_k} \varepsilon_{m_1 \ldots m_d}
```

### Left Complement

```math
\underline{\mathbf{B}}^{m_1 \ldots m_{d - k}} = \varepsilon_{m_1 \ldots m_d} B^{m_{d - k + 1} \ldots m_d}
```

### Grade Mapping

```math
\text{grade}(\overline{\mathbf{B}}) = d - \text{grade}(\mathbf{B})
```

### Double Complement

```math
\overline{\overline{\mathbf{B}}} = \pm \mathbf{B}
```

## Hodge Dual

The Hodge dual incorporates metric structure:

```math
(\star \mathbf{B})^{m_{k + 1} \ldots m_d} = \frac{1}{k!} B^{n_1 \ldots n_k} g_{n_1 m_1} \cdots g_{n_k m_k} \varepsilon^{m_1 \ldots m_d}
```

### Grade Mapping

```math
\text{grade}(\star \mathbf{B}) = d - \text{grade}(\mathbf{B})
```

### Examples

Scalar dual:

```math
\star s = s \mathbb{1}
```

where $\mathbb{1}$ is the pseudoscalar.

3D vector dual:

```math
\star \mathbf{v} = v^m \varepsilon^{mnp} \mathbf{e}_{np}
```

3D bivector dual:

```math
\star (\mathbf{e}_1 \wedge \mathbf{e}_2) = \pm \mathbf{e}_3
```

## Norms

### Squared Norm (Bilinear)

For grade-$k$ blade $\mathbf{B}$:

```math
|\mathbf{B}|^2 = \frac{1}{k!} B^{m_1 \ldots m_k} B^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}
```

The factorial prevents overcounting. For complex blades, this can return complex values.

### Hermitian Norm Squared (Sesquilinear)

For complex (phasor) blades, the Hermitian norm uses conjugation:

```math
|\mathbf{B}|^2_H = \frac{1}{k!} \overline{B^{m_1 \ldots m_k}} B^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}
```

This always returns real values for real metrics, giving the physical magnitude squared.

### Norm

```math
|\mathbf{B}| = \sqrt{||\mathbf{B}|^2|}
```

### Hermitian Norm

```math
|\mathbf{B}|_H = \sqrt{|\mathbf{B}|^2_H}
```

For phasors $\mathbf{v} = \mathbf{A} e^{i\phi}$, this gives the RMS amplitude $|\mathbf{A}|$.

### Properties

Unit basis vectors:

```math
|\mathbf{e}_m|^2 = g_{mm} = 1 \quad \text{(Euclidean)}
```

Pythagorean identity:

```math
|\mathbf{v}|^2 = \sum_{m} (v^m)^2 \quad \text{(Euclidean)}
```

### Normalization

```math
\hat{\mathbf{B}} = \frac{\mathbf{B}}{|\mathbf{B}|}
```

satisfies $|\hat{\mathbf{B}}| = 1$.

### Zero and Degenerate Cases

```math
|\mathbf{0}| = 0, \quad \text{unit}(\mathbf{0}) = \mathbf{0}
```

PGA ideal basis:

```math
|\mathbf{e}_0|^2 = g_{00} = 0
```

## Complex Conjugation

For complex blades representing phasors:

```math
\overline{\mathbf{B}}^{m_1 \ldots m_k} = \overline{B^{m_1 \ldots m_k}}
```

Conjugation acts on the coefficients only—the imaginary unit represents temporal phase, not geometric structure.

### Properties

```math
\overline{\overline{\mathbf{B}}} = \mathbf{B}
```

```math
\overline{\mathbf{A} + \mathbf{B}} = \overline{\mathbf{A}} + \overline{\mathbf{B}}
```

### Physical Applications

Time-averaged Poynting vector:

```math
\mathbf{S} = \frac{1}{2} \text{Re}(\mathbf{E} \wedge \overline{\mathbf{B}})
```

## Dot Product

For vectors:

```math
\mathbf{u} \cdot \mathbf{v} = g_{ab} u^a v^b
```

### Orthogonality

```math
\mathbf{e}_m \cdot \mathbf{e}_n = g_{mn} = \delta_{mn}
```

## Projections

### Projection Formula

```math
\text{proj}_{\mathbf{B}}(\mathbf{A}) = \frac{(\mathbf{A} \lrcorner \mathbf{B}) \lrcorner \mathbf{B}}{|\mathbf{B}|^2}
```

### Rejection Formula

```math
\text{rej}_{\mathbf{B}}(\mathbf{A}) = \mathbf{A} - \text{proj}_{\mathbf{B}}(\mathbf{A})
```

### Decomposition

```math
\mathbf{A} = \text{proj}_{\mathbf{B}}(\mathbf{A}) + \text{rej}_{\mathbf{B}}(\mathbf{A})
```

## Join and Meet

### Join (Union)

```math
\mathbf{A} \vee \mathbf{B} = \mathbf{A} \wedge \mathbf{B}
```

### Meet (Intersection)

```math
\mathbf{A} \wedge \mathbf{B} = \overline{\left(\overline{\mathbf{A}} \wedge \overline{\mathbf{B}}\right)}
```

### Grade Behavior

Join:

```math
\text{grade}(\mathbf{A} \vee \mathbf{B}) = j + k
```

Meet (transverse):

```math
\text{grade}(\mathbf{A} \wedge \mathbf{B}) = j + k - d
```

## Geometric Product

For vectors:

```math
\mathbf{u}\mathbf{v} = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \wedge \mathbf{v}
```

### Properties

Associativity:

```math
(\mathbf{u}\mathbf{v})\mathbf{w} = \mathbf{u}(\mathbf{v}\mathbf{w})
```

Vector contraction:

```math
\mathbf{v}^2 = \mathbf{v} \cdot \mathbf{v} = |\mathbf{v}|^2
```

Orthogonal anticommutativity:

```math
\mathbf{u} \perp \mathbf{v} \implies \mathbf{u}\mathbf{v} = -\mathbf{v}\mathbf{u}
```

### Bivector Squares

Unit bivectors in Euclidean space:

```math
\mathbf{e}_{12}^2 = \mathbf{e}_{23}^2 = \mathbf{e}_{31}^2 = -1
```

### Pseudoscalar Squares

```math
\mathbb{1}^2 = (-1)^{d(d - 1)/2}
```

- 2D, 3D: $\mathbb{1}^2 = -1$
- 4D: $\mathbb{1}^2 = +1$

## Reversion

For grade-$k$ blade:

```math
\widetilde{\mathbf{A}} = (-1)^{k(k - 1)/2} \mathbf{A}
```

Sign pattern:
- Grade 0, 1: $+$ (unchanged)
- Grade 2, 3: $-$ (sign flip)
- Grade 4, 5: $+$
- Grade 6, 7: $-$

### Properties

```math
\widetilde{\mathbf{A}\mathbf{B}} = \widetilde{\mathbf{B}} \widetilde{\mathbf{A}}
```

```math
\widetilde{\widetilde{\mathbf{A}}} = \mathbf{A}
```

## Inverse

For blade $\mathbf{u}$ with inverse $\mathbf{u}^{-1}$:

```math
\mathbf{u}^{-1}\mathbf{u} = \mathbf{u}\mathbf{u}^{-1} = 1
```

Vector inverse:

```math
\mathbf{v}^{-1} = \frac{\mathbf{v}}{|\mathbf{v}|^2}
```

Bivector inverse:

```math
\mathbf{B}^{-1} = \frac{\widetilde{\mathbf{B}}}{\mathbf{B}\widetilde{\mathbf{B}}}
```

## Outermorphisms

An outermorphism is a linear map $f: \bigwedge V \to \bigwedge W$ that preserves the wedge product:

```math
f(\mathbf{a} \wedge \mathbf{b}) = f(\mathbf{a}) \wedge f(\mathbf{b})
```

### Exterior Power

An outermorphism is completely determined by its action on grade-1 (vectors). If $A: V \to W$ is the linear map on vectors, the extension to grade-$k$ is the $k$-th exterior power $\bigwedge^k A$:

```math
(\bigwedge^k A)(\mathbf{v}_1 \wedge \cdots \wedge \mathbf{v}_k) = A(\mathbf{v}_1) \wedge \cdots \wedge A(\mathbf{v}_k)
```

### Component Form

For a $d \times d$ matrix $A^i{}_j$ and grade-$k$ blade $B^{m_1 \ldots m_k}$:

```math
(\bigwedge^k A)(\mathbf{B})^{i_1 \ldots i_k} = A^{i_1}{}_{m_1} \cdots A^{i_k}{}_{m_k} B^{m_1 \ldots m_k}
```

This is $k$ copies of $A$ contracting with $k$ blade indices.

### Scalars Invariant

Scalars (grade-0) are unchanged:

```math
(\bigwedge^0 A)(s) = s
```

### Determinant Property

The action on the pseudoscalar (grade-$d$) equals multiplication by the determinant:

```math
(\bigwedge^d A)(\mathbb{1}) = \det(A) \cdot \mathbb{1}
```

### Composition

For outermorphisms $f$ and $g$ with vector maps $A$ and $B$:

```math
(f \circ g)|_{\text{grade-}k} = \bigwedge^k(AB)
```

Composition of outermorphisms corresponds to matrix multiplication of their vector maps.
