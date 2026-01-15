# Operations: Mathematical Properties

Mathematical foundations for wedge product, interior product, duality, norms, and projections.

## Wedge Product

The wedge (exterior) product constructs higher-grade blades.

### Definition

For grade-$j$ blade $\mathbf{A}$ and grade-$k$ blade $\mathbf{B}$:

$$
(\mathbf{A} \wedge \mathbf{B})^{m_1 \ldots m_{j+k}} = \frac{1}{j! \, k!} A^{m_1 \ldots m_j} B^{m_{j+1} \ldots m_{j+k}} \times \text{antisymmetrize}
$$

Result has grade $(j + k)$.

### Anticommutativity

For vectors:

$$
\mathbf{u} \wedge \mathbf{v} = -\mathbf{v} \wedge \mathbf{u}
$$

General form:

$$
\mathbf{A} \wedge \mathbf{B} = (-1)^{jk} \mathbf{B} \wedge \mathbf{A}
$$

### Nilpotency

$$
\mathbf{v} \wedge \mathbf{v} = 0
$$

Linear dependence:

$$
\mathbf{u} = \alpha \mathbf{v} \implies \mathbf{u} \wedge \mathbf{v} = 0
$$

### Associativity

$$
(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})
$$

### Grade Calculation

$$
\text{grade}(\mathbf{A} \wedge \mathbf{B}) = \text{grade}(\mathbf{A}) + \text{grade}(\mathbf{B})
$$

## Interior Product

The interior product (left contraction) contracts indices using the metric.

### Definition

For grade-$j$ blade $\mathbf{A}$ and grade-$k$ blade $\mathbf{B}$ with $j \leq k$:

$$
(\mathbf{A} \lrcorner \mathbf{B})^{n_1 \ldots n_{k-j}} = A^{m_1 \ldots m_j} B_{m_1 \ldots m_j}^{n_1 \ldots n_{k-j}}
$$

Indices lowered via metric:

$$
B_{m_1 \ldots m_j}^{n_1 \ldots n_{k-j}} = g_{m_1 p_1} \cdots g_{m_j p_j} B^{p_1 \ldots p_j n_1 \ldots n_{k-j}}
$$

### Grade Reduction

$$
\text{grade}(\mathbf{A} \lrcorner \mathbf{B}) = \text{grade}(\mathbf{B}) - \text{grade}(\mathbf{A})
$$

### Special Cases

When $j > k$:

$$
\mathbf{A} \lrcorner \mathbf{B} = 0
$$

Scalar contraction:

$$
s \lrcorner \mathbf{B} = s \mathbf{B}
$$

### Orthogonality

For orthogonal basis vectors $\mathbf{e}_m \perp \mathbf{e}_n$ with $m \neq n$:

$$
\mathbf{e}_m \lrcorner (\mathbf{e}_n \wedge \mathbf{e}_p) = 0 \quad \text{when } m \notin \{n, p\}
$$

## Complement Operations

Complement operations map between dual grades using the Levi-Civita symbol.

### Right Complement

For grade-$k$ blade in $d$ dimensions:

$$
\overline{\mathbf{B}}^{m_{k+1} \ldots m_d} = B^{m_1 \ldots m_k} \varepsilon_{m_1 \ldots m_d}
$$

### Left Complement

$$
\underline{\mathbf{B}}^{m_1 \ldots m_{d-k}} = \varepsilon_{m_1 \ldots m_d} B^{m_{d-k+1} \ldots m_d}
$$

### Grade Mapping

$$
\text{grade}(\overline{\mathbf{B}}) = d - \text{grade}(\mathbf{B})
$$

### Double Complement

$$
\overline{\overline{\mathbf{B}}} = \pm \mathbf{B}
$$

## Hodge Dual

The Hodge dual incorporates metric structure:

$$
(\star \mathbf{B})^{m_{k+1} \ldots m_d} = \frac{1}{k!} B^{n_1 \ldots n_k} g_{n_1 m_1} \cdots g_{n_k m_k} \varepsilon^{m_1 \ldots m_d}
$$

### Grade Mapping

$$
\text{grade}(\star \mathbf{B}) = d - \text{grade}(\mathbf{B})
$$

### Examples

Scalar dual:

$$
\star s = s \, \mathbb{1}
$$

where $\mathbb{1}$ is the pseudoscalar.

3D vector dual:

$$
\star \mathbf{v} = v^m \varepsilon^{mnp} \mathbf{e}_{np}
$$

3D bivector dual:

$$
\star (\mathbf{e}_1 \wedge \mathbf{e}_2) = \pm \mathbf{e}_3
$$

## Norms

### Squared Norm

For grade-$k$ blade $\mathbf{B}$:

$$
|\mathbf{B}|^2 = \frac{1}{k!} B^{m_1 \ldots m_k} B^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}
$$

The factorial prevents overcounting.

### Norm

$$
|\mathbf{B}| = \sqrt{||\mathbf{B}|^2|}
$$

### Properties

Unit basis vectors:

$$
|\mathbf{e}_m|^2 = g_{mm} = 1 \quad \text{(Euclidean)}
$$

Pythagorean identity:

$$
|\mathbf{v}|^2 = \sum_{m} (v^m)^2 \quad \text{(Euclidean)}
$$

### Normalization

$$
\hat{\mathbf{B}} = \frac{\mathbf{B}}{|\mathbf{B}|}
$$

satisfies $|\hat{\mathbf{B}}| = 1$.

### Zero and Degenerate Cases

$$
|\mathbf{0}| = 0, \quad \text{normalize}(\mathbf{0}) = \mathbf{0}
$$

PGA ideal basis:

$$
|\mathbf{e}_0|^2 = g_{00} = 0
$$

## Dot Product

For vectors:

$$
\mathbf{u} \cdot \mathbf{v} = g_{ab} u^a v^b
$$

### Orthogonality

$$
\mathbf{e}_m \cdot \mathbf{e}_n = g_{mn} = \delta_{mn}
$$

## Projections

### Projection Formula

$$
\text{proj}_{\mathbf{B}}(\mathbf{A}) = \frac{(\mathbf{A} \lrcorner \mathbf{B}) \lrcorner \mathbf{B}}{|\mathbf{B}|^2}
$$

### Rejection Formula

$$
\text{rej}_{\mathbf{B}}(\mathbf{A}) = \mathbf{A} - \text{proj}_{\mathbf{B}}(\mathbf{A})
$$

### Decomposition

$$
\mathbf{A} = \text{proj}_{\mathbf{B}}(\mathbf{A}) + \text{rej}_{\mathbf{B}}(\mathbf{A})
$$

## Join and Meet

### Join (Union)

$$
\mathbf{A} \vee \mathbf{B} = \mathbf{A} \wedge \mathbf{B}
$$

### Meet (Intersection)

$$
\mathbf{A} \wedge \mathbf{B} = \overline{\left(\overline{\mathbf{A}} \wedge \overline{\mathbf{B}}\right)}
$$

### Grade Behavior

Join:

$$
\text{grade}(\mathbf{A} \vee \mathbf{B}) = j + k
$$

Meet (transverse):

$$
\text{grade}(\mathbf{A} \wedge \mathbf{B}) = j + k - d
$$

## Geometric Product

For vectors:

$$
\mathbf{u}\mathbf{v} = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \wedge \mathbf{v}
$$

### Properties

Associativity:

$$
(\mathbf{u}\mathbf{v})\mathbf{w} = \mathbf{u}(\mathbf{v}\mathbf{w})
$$

Vector contraction:

$$
\mathbf{v}^2 = \mathbf{v} \cdot \mathbf{v} = |\mathbf{v}|^2
$$

Orthogonal anticommutativity:

$$
\mathbf{u} \perp \mathbf{v} \implies \mathbf{u}\mathbf{v} = -\mathbf{v}\mathbf{u}
$$

### Bivector Squares

Unit bivectors in Euclidean space:

$$
\mathbf{e}_{12}^2 = \mathbf{e}_{23}^2 = \mathbf{e}_{31}^2 = -1
$$

### Pseudoscalar Squares

$$
\mathbb{1}^2 = (-1)^{d(d-1)/2}
$$

- 2D, 3D: $\mathbb{1}^2 = -1$
- 4D: $\mathbb{1}^2 = +1$

## Reversion

For grade-$k$ blade:

$$
\widetilde{\mathbf{A}} = (-1)^{k(k-1)/2} \mathbf{A}
$$

Sign pattern:
- Grade 0, 1: $+$ (unchanged)
- Grade 2, 3: $-$ (sign flip)
- Grade 4, 5: $+$
- Grade 6, 7: $-$

### Properties

$$
\widetilde{\mathbf{A}\mathbf{B}} = \widetilde{\mathbf{B}} \, \widetilde{\mathbf{A}}
$$

$$
\widetilde{\widetilde{\mathbf{A}}} = \mathbf{A}
$$

## Inverse

For blade $\mathbf{u}$ with inverse $\mathbf{u}^{-1}$:

$$
\mathbf{u}^{-1}\mathbf{u} = \mathbf{u}\mathbf{u}^{-1} = 1
$$

Vector inverse:

$$
\mathbf{v}^{-1} = \frac{\mathbf{v}}{|\mathbf{v}|^2}
$$

Bivector inverse:

$$
\mathbf{B}^{-1} = \frac{\widetilde{\mathbf{B}}}{\mathbf{B}\widetilde{\mathbf{B}}}
$$
