# Unit Tests: Mathematical Specification

This document provides a mathematical description of all unit tests implemented for morphis. Each test verifies fundamental algebraic properties, geometric relationships, or computational correctness of the implemented operations.

## Data Model Tests

### Metric Construction and Properties

The metric tensor $g_{ab}$ defines the inner product structure of the vector space. For projective geometric algebra (PGA), the metric exhibits degeneracy in the ideal direction.

**PGA Metric Structure**

The PGA metric for $d$-dimensional Euclidean space embedded in $(d + 1)$-dimensional projective space satisfies:

$$g = \text{diag}(0, 1, 1, \ldots, 1)$$

with signature $(0, 1, 1, \ldots, 1)$ where the first component corresponds to the ideal point $e_0$ and satisfies:

$$g_{00} = 0$$

$$g_{mm} = 1 \quad \text{for } m \geq 1$$

The dimension relationship follows as $\dim(g) = d + 1$ for $d$-dimensional Euclidean space.

**Euclidean Metric**

The standard Euclidean metric in $d$ dimensions is the identity matrix:

$$g = \mathbb{I}_d = \text{diag}(1, 1, \ldots, 1)$$

with signature $(1, 1, \ldots, 1)$ and $\dim(g) = d$.

**Metric Indexing**

Direct component access verifies the algebraic structure:

$$g_{12} = 0 \quad \text{(off-diagonal vanishes)}$$

$$g[1:3, 1:3] = \mathbb{I}_2 \quad \text{(Euclidean subblock)}$$

### Vector Construction and Shape Validation

A grade-$k$ Vector in $d$-dimensional space has components $B^{m_1 \ldots m_k}$ satisfying antisymmetry:

$$B^{\ldots m \ldots n \ldots} = -B^{\ldots n \ldots m \ldots}$$

The storage shape follows as $(*\text{lot}, d, d, \ldots, d)$ with $k$ trailing geometric dimensions.

**Grade-Specific Construction**

For grade-1 vectors:

$$\mathbf{v} = v^m \mathbf{e}_m$$

stored as shape $(d,)$ with $\dim(\mathbf{v}) = d$.

For bivectors (grade 2):

$$\mathbf{B} = B^{mn} \mathbf{e}_{mn}$$

stored as shape $(d, d)$ with antisymmetric components.

For trivectors (grade 3):

$$\mathbf{T} = T^{mnp} \mathbf{e}_{mnp}$$

stored as shape $(d, d, d)$ with full antisymmetry over all indices.

**Collection Dimensions**

Batch processing via lot dimensions allows shape $(*\text{lot}, *\text{geo})$ where:

$$\text{cdim} = \text{number of leading batch dimensions}$$

$$\text{ndim} = \text{cdim} + \text{grade}$$

For batch shape $(N,)$ with vectors in $\mathbb{R}^d$, storage is $(N, d)$ with $\text{cdim} = 1$.

**Shape Validation**

The implementation enforces:

$$\text{grade} \geq 0$$

$$\text{cdim} \geq 0$$

$$\text{ndim} = \text{cdim} + \text{grade}$$

$$\text{shape}[-k] = d \quad \text{for all } k \in [1, \text{grade}]$$

Violations raise `ValueError` with descriptive messages identifying the constraint failure.

### Vector Arithmetic

Vectors of equal grade form a vector space under addition and scalar multiplication.

**Addition**

For Vectors $\mathbf{A}$, $\mathbf{B}$ of grade $k$:

$$(\mathbf{A} + \mathbf{B})^{m_1 \ldots m_k} = A^{m_1 \ldots m_k} + B^{m_1 \ldots m_k}$$

This operation preserves grade and dimension:

$$\text{grade}(\mathbf{A} + \mathbf{B}) = \text{grade}(\mathbf{A}) = \text{grade}(\mathbf{B})$$

$$\dim(\mathbf{A} + \mathbf{B}) = \dim(\mathbf{A}) = \dim(\mathbf{B})$$

Collection dimensions broadcast according to NumPy rules.

**Subtraction**

The difference of two Vectors follows:

$$(\mathbf{A} - \mathbf{B})^{m_1 \ldots m_k} = A^{m_1 \ldots m_k} - B^{m_1 \ldots m_k}$$

**Scalar Multiplication**

For scalar $\alpha$ and Vector $\mathbf{B}$:

$$(\alpha \mathbf{B})^{m_1 \ldots m_k} = \alpha B^{m_1 \ldots m_k}$$

This satisfies commutativity: $\alpha \mathbf{B} = \mathbf{B} \alpha$.

**Scalar Division**

Division by nonzero scalar $\alpha$:

$$\left(\frac{\mathbf{B}}{\alpha}\right)^{m_1 \ldots m_k} = \frac{B^{m_1 \ldots m_k}}{\alpha}$$

**Negation**

The additive inverse:

$$(-\mathbf{B})^{m_1 \ldots m_k} = -B^{m_1 \ldots m_k}$$

satisfies $\mathbf{B} + (-\mathbf{B}) = 0$.

**Type Checking**

Grade mismatch raises `ValueError`:

$$\text{grade}(\mathbf{A}) \neq \text{grade}(\mathbf{B}) \implies \mathbf{A} + \mathbf{B} \text{ undefined}$$

Dimension mismatch similarly fails:

$$\dim(\mathbf{A}) \neq \dim(\mathbf{B}) \implies \mathbf{A} + \mathbf{B} \text{ undefined}$$

### MultiVector Operations

A multivector combines Vectors of different grades:

$$\mathbf{M} = \sum_{k=0}^{d} \mathbf{M}_{\langle k \rangle}$$

where $\mathbf{M}_{\langle k \rangle}$ denotes the grade-$k$ component.

**Grade Selection**

For multivector $\mathbf{M}$, the grade projection:

$$\langle \mathbf{M} \rangle_k = \mathbf{M}_{\langle k \rangle}$$

extracts the grade-$k$ component, returning `None` if that grade is absent.

**Multivector Addition**

For multivectors $\mathbf{M}$, $\mathbf{N}$:

$$(\mathbf{M} + \mathbf{N})_{\langle k \rangle} = \mathbf{M}_{\langle k \rangle} + \mathbf{N}_{\langle k \rangle}$$

where each grade adds independently.

**Construction from Vectors**

Given Vectors $\mathbf{B}_1, \ldots, \mathbf{B}_n$:

$$\mathbf{M} = \sum_{k} \mathbf{B}_k$$

Duplicate grades sum:

$$\mathbf{M}_{\langle k \rangle} = \sum_{i: \text{grade}(\mathbf{B}_i) = k} \mathbf{B}_i$$

**Validation**

All component Vectors must satisfy:

$$\dim(\mathbf{B}_k) = d \quad \text{for all } k$$

$$\text{cdim}(\mathbf{B}_k) = c \quad \text{for all } k$$

$$\text{grade}(\mathbf{B}_k) = k \quad \text{(key matches grade)}$$

## Operations Tests

### Wedge Product

The wedge (exterior) product constructs higher-grade Vectors from lower-grade inputs.

**Fundamental Formula**

For grade-$j$ Vector $\mathbf{A}$ and grade-$k$ Vector $\mathbf{B}$:

$$(\mathbf{A} \wedge \mathbf{B})^{m_1 \ldots m_{j + k}} = \frac{1}{j! \, k!} A^{m_1 \ldots m_j} B^{m_{j + 1} \ldots m_{j + k}} \times \text{antisymmetrize}$$

The result has grade $(j + k)$.

**Anticommutativity**

For grade-1 vectors $\mathbf{u}$, $\mathbf{v}$:

$$\mathbf{u} \wedge \mathbf{v} = -\mathbf{v} \wedge \mathbf{u}$$

More generally, for grades $j$ and $k$:

$$\mathbf{A} \wedge \mathbf{B} = (-1)^{jk} \mathbf{B} \wedge \mathbf{A}$$

**Nilpotency**

Any vector wedged with itself vanishes:

$$\mathbf{v} \wedge \mathbf{v} = 0$$

This extends to linear dependence:

$$\mathbf{u} = \alpha \mathbf{v} \implies \mathbf{u} \wedge \mathbf{v} = 0$$

**Associativity**

The wedge product associates:

$$(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})$$

This allows construction of grade-$k$ Vectors via sequential wedging:

$$\mathbf{T} = \mathbf{e}_1 \wedge \mathbf{e}_2 \wedge \mathbf{e}_3$$

**Grade Calculation**

The grade of a wedge product follows:

$$\text{grade}(\mathbf{A} \wedge \mathbf{B}) = \text{grade}(\mathbf{A}) + \text{grade}(\mathbf{B})$$

For three vectors:

$$\text{grade}(\mathbf{a} \wedge \mathbf{b} \wedge \mathbf{c}) = 3$$

**Batch Operations**

Collection dimensions broadcast properly:

- Batch $\times$ single: output has batch shape
- Batch $\times$ batch: pointwise product with matching shapes

### Interior Product

The interior product (left contraction) contracts indices using the metric.

**Definition**

For grade-$j$ Vector $\mathbf{A}$ and grade-$k$ Vector $\mathbf{B}$ with $j \leq k$:

$$(\mathbf{A} \lrcorner \mathbf{B})^{n_1 \ldots n_{k - j}} = A^{m_1 \ldots m_j} B_{m_1 \ldots m_j}^{n_1 \ldots n_{k - j}}$$

where indices are lowered via the metric:

$$B_{m_1 \ldots m_j}^{n_1 \ldots n_{k - j}} = g_{m_1 p_1} \cdots g_{m_j p_j} B^{p_1 \ldots p_j n_1 \ldots n_{k - j}}$$

The result has grade $(k - j)$.

**Grade Reduction**

Each contraction reduces grade by the grade of the first argument:

$$\text{grade}(\mathbf{A} \lrcorner \mathbf{B}) = \text{grade}(\mathbf{B}) - \text{grade}(\mathbf{A})$$

**Zero Cases**

When $j > k$:

$$\mathbf{A} \lrcorner \mathbf{B} = 0$$

**Scalar Cases**

For scalar $s$ (grade 0):

$$s \lrcorner \mathbf{B} = s \mathbf{B}$$

**Orthogonality**

For orthogonal basis vectors $\mathbf{e}_m \perp \mathbf{e}_n$ with $m \neq n$:

$$\mathbf{e}_m \lrcorner (\mathbf{e}_n \wedge \mathbf{e}_p) = 0$$

when $m \notin \{n, p\}$.

**Projection Formula**

For vector $\mathbf{e}_1$ and bivector $\mathbf{e}_1 \wedge \mathbf{e}_2$:

$$\mathbf{e}_1 \lrcorner (\mathbf{e}_1 \wedge \mathbf{e}_2) = \pm \mathbf{e}_2$$

with sign determined by orientation.

### Complement Operations

Complement operations map between dual grades using the Levi-Civita symbol.

**Right Complement**

For grade-$k$ Vector in $d$ dimensions:

$$\overline{\mathbf{B}}^{m_{k + 1} \ldots m_d} = B^{m_1 \ldots m_k} \varepsilon_{m_1 \ldots m_d}$$

This maps grade $k$ to grade $(d - k)$:

$$\text{grade}(\overline{\mathbf{B}}) = d - \text{grade}(\mathbf{B})$$

**Left Complement**

The left complement:

$$\underline{\mathbf{B}}^{m_1 \ldots m_{d - k}} = \varepsilon_{m_1 \ldots m_d} B^{m_{d - k + 1} \ldots m_d}$$

also maps grade $k$ to grade $(d - k)$.

**Dimensional Behavior**

In 3D, vectors map to bivectors:

$$\text{grade}(\overline{\mathbf{v}}) = 3 - 1 = 2$$

In 4D, bivectors map to bivectors:

$$\text{grade}(\overline{\mathbf{B}}) = 4 - 2 = 2$$

**Double Complement**

Applying complement twice recovers the original (up to sign):

$$\overline{\overline{\mathbf{B}}} = \pm \mathbf{B}$$

### Hodge Dual

The Hodge dual incorporates the metric structure:

$$(\star \mathbf{B})^{m_{k + 1} \ldots m_d} = \frac{1}{k!} B^{n_1 \ldots n_k} g_{n_1 m_1} \cdots g_{n_k m_k} \varepsilon^{m_1 \ldots m_d}$$

**Grade Mapping**

The Hodge dual maps grade $k$ to grade $(d - k)$:

$$\text{grade}(\star \mathbf{B}) = d - \text{grade}(\mathbf{B})$$

**Scalar Dual**

For scalar $s$ in $d$ dimensions:

$$\star s = s \, \mathbb{1}$$

where $\mathbb{1}$ is the pseudoscalar (grade $d$).

**Vector Dual in 3D**

In 3D Euclidean space, the dual of a vector is a bivector:

$$\star \mathbf{v} = v^m \varepsilon^{mnp} \mathbf{e}_{np}$$

This creates the familiar cross-product-like correspondence.

**Bivector Dual in 3D**

The dual of a bivector in 3D gives a vector:

$$\star (\mathbf{e}_1 \wedge \mathbf{e}_2) = \pm \mathbf{e}_3$$

establishing the vector-bivector duality in three dimensions.

### Norms

The norm of a Vector incorporates the metric structure and factorial normalization.

**Squared Norm**

For grade-$k$ Vector $\mathbf{B}$:

$$|\mathbf{B}|^2 = \frac{1}{k!} B^{m_1 \ldots m_k} B^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}$$

The factorial prevents overcounting from antisymmetric components.

**Norm**

The norm is the square root of absolute value:

$$|\mathbf{B}| = \sqrt{||\mathbf{B}|^2|}$$

**Unit Vectors**

For Euclidean basis vector $\mathbf{e}_m$:

$$|\mathbf{e}_m|^2 = g_{mm} = 1$$

$$|\mathbf{e}_m| = 1$$

**Pythagorean Identity**

For vector $\mathbf{v} = v^m \mathbf{e}_m$ in Euclidean space:

$$|\mathbf{v}|^2 = g_{mn} v^m v^n = \sum_{m} (v^m)^2$$

giving the standard Pythagorean formula.

**Unit Bivector**

For $\mathbf{B} = \mathbf{e}_1 \wedge \mathbf{e}_2$ in Euclidean space:

$$|\mathbf{B}|^2 > 0$$

reflecting positive-definite area.

**Unit Vectors**

The unit vector:

$$\hat{\mathbf{B}} = \frac{\mathbf{B}}{|\mathbf{B}|}$$

satisfies $|\hat{\mathbf{B}}| = 1$.

**Zero Vector Handling**

For zero Vector:

$$|\mathbf{0}| = 0$$

$$\text{unit}(\mathbf{0}) = \mathbf{0}$$

**PGA Metric Degeneracy**

In PGA with metric $g = \text{diag}(0, 1, 1, \ldots)$:

$$|\mathbf{e}_0|^2 = g_{00} = 0$$

$$|\mathbf{e}_0| = 0$$

The ideal basis vector has zero norm, reflecting the projective structure.

### Join and Meet

Join and meet are dual operations connecting union and intersection of subspaces.

**Join as Wedge**

The join of two Vectors is their wedge product:

$$\mathbf{A} \vee \mathbf{B} = \mathbf{A} \wedge \mathbf{B}$$

This constructs the smallest subspace containing both.

**Meet via Duality**

The meet uses complement duality:

$$\mathbf{A} \wedge \mathbf{B} = \overline{\left(\overline{\mathbf{A}} \wedge \overline{\mathbf{B}}\right)}$$

**Grade Behavior**

For grade-$j$ and grade-$k$ Vectors:

$$\text{grade}(\mathbf{A} \vee \mathbf{B}) = j + k$$

The meet grade depends on intersection dimension:

$$\text{grade}(\mathbf{A} \wedge \mathbf{B}) = j + k - d$$

when the Vectors span transverse subspaces.

**Plane Intersection**

In 4D, two planes (trivectors) meet in a line:

$$\text{grade}(\text{plane}_1 \wedge \text{plane}_2) = 3 + 3 - 4 = 2$$

**Plane-Line Intersection**

A plane and line meet in a point:

$$\text{grade}(\text{plane} \wedge \text{line}) = 3 + 2 - 4 = 1$$

### Dot Product

The dot product for vectors uses the metric directly.

**Formula**

For vectors $\mathbf{u}$, $\mathbf{v}$:

$$\mathbf{u} \cdot \mathbf{v} = g_{ab} u^a v^b$$

**Orthogonality**

For orthogonal basis vectors:

$$\mathbf{e}_m \cdot \mathbf{e}_n = g_{mn} = \delta_{mn}$$

giving:

$$\mathbf{e}_m \cdot \mathbf{e}_n = 0 \quad \text{when } m \neq n$$

**Self-Product**

For unit vector $\mathbf{e}_m$:

$$\mathbf{e}_m \cdot \mathbf{e}_m = g_{mm} = 1$$

**General Vectors**

For $\mathbf{u} = (1, 2, 3, 4)$ and $\mathbf{v} = (2, 1, 1, 1)$ in Euclidean space:

$$\mathbf{u} \cdot \mathbf{v} = 1 \times 2 + 2 \times 1 + 3 \times 1 + 4 \times 1 = 11$$

### Projections

Projection and rejection decompose a Vector relative to another.

**Projection Formula**

The projection of $\mathbf{A}$ onto $\mathbf{B}$:

$$\text{proj}_{\mathbf{B}}(\mathbf{A}) = \frac{(\mathbf{A} \lrcorner \mathbf{B}) \lrcorner \mathbf{B}}{|\mathbf{B}|^2}$$

**Rejection Formula**

The rejection (orthogonal component):

$$\text{rej}_{\mathbf{B}}(\mathbf{A}) = \mathbf{A} - \text{proj}_{\mathbf{B}}(\mathbf{A})$$

**Orthogonal Decomposition**

Projection and rejection satisfy:

$$\mathbf{A} = \text{proj}_{\mathbf{B}}(\mathbf{A}) + \text{rej}_{\mathbf{B}}(\mathbf{A})$$

**Vector Projection**

For $\mathbf{v} = (1, 1, 0, 0)$ onto $\mathbf{e}_1 = (1, 0, 0, 0)$:

$$\text{proj}_{\mathbf{e}_1}(\mathbf{v}) = (1, 0, 0, 0)$$

$$\text{rej}_{\mathbf{e}_1}(\mathbf{v}) = (0, 1, 0, 0)$$

## Exponential and Logarithm Tests

### Bivector Exponential

For bivector $\mathbf{B}$ where $\mathbf{B}^2 = \lambda$ is scalar:

$$e^{\mathbf{B}} = \begin{cases}
\cos\sqrt{-\lambda} + \frac{\mathbf{B}}{\sqrt{-\lambda}} \sin\sqrt{-\lambda} & \text{if } \lambda < 0 \\
\cosh\sqrt{\lambda} + \frac{\mathbf{B}}{\sqrt{\lambda}} \sinh\sqrt{\lambda} & \text{if } \lambda > 0 \\
1 + \mathbf{B} & \text{if } \lambda = 0
\end{cases}$$

**Unit Bivector**

For unit bivector $\hat{\mathbf{B}}$ with $\hat{\mathbf{B}}^2 = -1$:

$$e^{\theta \hat{\mathbf{B}}} = \cos\theta + \hat{\mathbf{B}}\sin\theta$$

**Rotor Construction**

A rotor for rotation by angle $\theta$ in the plane of unit bivector $\hat{\mathbf{B}}$:

$$R = e^{-\hat{\mathbf{B}}\theta/2} = \cos(\theta/2) - \hat{\mathbf{B}}\sin(\theta/2)$$

**Rotor Normalization**

Rotors satisfy $R\tilde{R} = 1$:

$$|R|^2 = \cos^2(\theta/2) + \sin^2(\theta/2) = 1$$

### Versor Logarithm

For rotor $R = a + \mathbf{B}$ where $a$ is scalar and $\mathbf{B}$ is bivector:

$$\log R = \arctan2(|\mathbf{B}|, a) \cdot \frac{\mathbf{B}}{|\mathbf{B}|}$$

**Inverse Property**

$$e^{\log R} = R$$

$$\log(e^{\mathbf{B}}) = \mathbf{B} \quad \text{(for small } |\mathbf{B}| \text{)}$$

### Slerp

Spherical linear interpolation between rotors $R_0$ and $R_1$:

$$R(t) = R_0 \, e^{t \log(R_0^{-1} R_1)}$$

**Boundary Conditions**

$$R(0) = R_0$$
$$R(1) = R_1$$

**Midpoint**

$$R(0.5) = R_0 \, e^{0.5 \log(R_0^{-1} R_1)}$$

## Utilities Tests

### Permutation Sign

The sign of a permutation indicates parity.

**Definition**

For permutation $\sigma$:

$$\text{sgn}(\sigma) = \begin{cases}
+1 & \text{if } \sigma \text{ is even} \\
-1 & \text{if } \sigma \text{ is odd}
\end{cases}$$

**Identity**

The identity permutation is even:

$$\text{sgn}((0, 1, 2)) = +1$$

**Single Transposition**

Swapping two elements is odd:

$$\text{sgn}((1, 0, 2)) = -1$$

**Cyclic Permutations in 3D**

Forward cycle $(1, 2, 0)$ is even:

$$\text{sgn}((1, 2, 0)) = +1$$

Backward cycle $(2, 1, 0)$ is odd:

$$\text{sgn}((2, 1, 0)) = -1$$

### Antisymmetrization

Antisymmetrization projects tensors onto the antisymmetric subspace.

**Formula**

For rank-$k$ tensor $T$:

$$T^{[m_1 \ldots m_k]} = \sum_{\sigma \in S_k} \text{sgn}(\sigma) \, T^{m_{\sigma(1)} \ldots m_{\sigma(k)}}$$

(without $1/k!$ normalization).

**Antisymmetry Property**

The result satisfies:

$$T^{[m_1 \ldots m_k]}_{[n_1 \ldots n_k]} = -T^{[n_1 \ldots m_k]}_{[m_1 \ldots n_k]}$$

for any index swap.

**Diagonal Vanishing**

Antisymmetric components with repeated indices vanish:

$$T^{[m m n]} = 0$$

### Levi-Civita Tensor

The Levi-Civita symbol is the fully antisymmetric tensor.

**Definition**

For $d$ dimensions:

$$\varepsilon^{m_1 \ldots m_d} = \begin{cases}
+1 & \text{if } (m_1, \ldots, m_d) \text{ is even permutation of } (0, 1, \ldots, d-1) \\
-1 & \text{if odd permutation} \\
0 & \text{if any indices repeat}
\end{cases}$$

**Contraction Identity**

Self-contraction gives the factorial:

$$\varepsilon^{abc} \varepsilon_{abc} = d! = 6 \quad \text{for } d = 3$$

### Generalized Kronecker Delta

The generalized delta antisymmetrizes under contraction.

**Definition**

$$\delta^{m_1 \ldots m_k}_{n_1 \ldots n_k} = \frac{1}{k!} \sum_{\sigma \in S_k} \text{sgn}(\sigma) \, \delta^{m_1}_{n_{\sigma(1)}} \cdots \delta^{m_k}_{n_{\sigma(k)}}$$

**Contraction**

Self-contraction counts combinations:

$$\delta^{ab}_{ab} = \binom{d}{k}$$

For $k = 2$, $d = 3$:

$$\delta^{ab}_{ab} = \binom{3}{2} = 3$$

## Projective Operations Tests

### Point and Direction Embedding

Projective geometric algebra embeds Euclidean geometry via homogeneous coordinates.

**Point Embedding**

A Euclidean point $\mathbf{x} \in \mathbb{R}^d$ embeds as:

$$\mathbf{p} = e_0 + x^m e_m$$

with weight 1 in the ideal component.

**Direction Embedding**

A Euclidean direction $\mathbf{v} \in \mathbb{R}^d$ embeds as:

$$\mathbf{d} = v^m e_m$$

with zero weight, representing a point at infinity.

**Weight Extraction**

The weight extracts the ideal component:

$$w(\mathbf{p}) = p^0$$

Points satisfy $w(\mathbf{p}) = 1$, directions satisfy $w(\mathbf{d}) = 0$.

**Bulk Extraction**

The bulk extracts Euclidean components:

$$\text{bulk}(\mathbf{p}) = (p^1, p^2, \ldots, p^d)$$

**Euclidean Projection**

For points with $w \neq 0$:

$$\text{euclidean}(\mathbf{p}) = \frac{\text{bulk}(\mathbf{p})}{w(\mathbf{p})}$$

### Geometric Constructors

PGA constructs higher-grade objects via wedge products.

**Line through Two Points**

$$\ell = \mathbf{p} \wedge \mathbf{q}$$

This bivector represents the line through points $\mathbf{p}$ and $\mathbf{q}$.

**Plane through Three Points**

$$\pi = \mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r}$$

This trivector represents the plane containing $\mathbf{p}$, $\mathbf{q}$, $\mathbf{r}$.

### Distance Functions

Distances use norms of wedge products normalized by object norms.

**Point-to-Point Distance**

For points $\mathbf{p}$, $\mathbf{q}$:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{g_{mn} (x_q^m - x_p^m)(x_q^n - x_p^n)}$$

**Point-to-Line Distance**

$$d(\mathbf{p}, \ell) = \frac{|\mathbf{p} \wedge \ell|}{|\ell|}$$

**Point-to-Plane Distance**

$$d(\mathbf{p}, \pi) = \frac{|\mathbf{p} \wedge \pi|}{|\pi|}$$

### Incidence Predicates

Incidence tests verify geometric relationships via vanishing wedge products.

**Collinearity**

Three points $\mathbf{p}$, $\mathbf{q}$, $\mathbf{r}$ are collinear if:

$$\mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r} = 0$$

**Point on Line**

Point $\mathbf{p}$ lies on line $\ell$ if:

$$\mathbf{p} \wedge \ell = 0$$

**Point on Plane**

Point $\mathbf{p}$ lies on plane $\pi$ if:

$$\mathbf{p} \wedge \pi = 0$$

**Tolerance**

All incidence tests use tolerance $\epsilon = 10^{-10}$:

$$||\mathbf{A} \wedge \mathbf{B}|| < \epsilon$$

## Integration Tests

### Meet-Join Duality

Complement duality relates meet and join:

$$\overline{\mathbf{A}} \wedge \overline{\mathbf{B}} = \overline{(\mathbf{A} \vee \mathbf{B})}$$

### Geometric Construction Consistency

Points constructed as embeddings satisfy incidence with derived objects:

$$\mathbf{p} \in \ell \implies \mathbf{p} \wedge \ell = 0$$

where $\ell = \mathbf{p} \wedge \mathbf{q}$.

## Edge Cases

### Zero Vector Handling

Zero Vectors satisfy:

$$|\mathbf{0}| = 0$$

$$\text{unit}(\mathbf{0}) = \mathbf{0}$$

### Complex Dtype Support

All operations support complex coefficients:

$$\mathbf{v} = (1 + i, 0, 0, 0)$$

$$\mathbf{v} \wedge \mathbf{w} \text{ preserves complex structure}$$

### Broadcasting

Collection dimensions broadcast according to NumPy rules across all operations, preserving geometric semantics while enabling batch processing.
