# Operations: Mathematical Properties

Mathematical foundations for wedge product, interior product, duality, norms, and projections.

`operations/tests/test_identities.py` checks these for every grade, in Euclidean, spacetime (both sign conventions), complex, and projective metrics. It uses random k-vectors, and random blades not aligned with the axes where an identity holds only for blades. Its reference for the geometric product is a multiplication table of basis blades built from the metric alone, and every other operation is tied to the geometric product by the identity that defines it.

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
\mathbf{u} = α \mathbf{v} \implies \mathbf{u} \wedge \mathbf{v} = 0
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

$$
\begin{aligned}
A \lrcorner B &= \langle A B \rangle_{k - j} \\
(A \lrcorner B)^{n_1 \ldots n_{k - j}} &= \frac{1}{j!} A^{m_j \ldots m_1} B_{m_1 \ldots m_j}^{\ \ \ \ \ \ \ \ n_1 \ldots n_{k - j}}
\end{aligned}
$$

so that $\mathbf e_{12} \lrcorner \mathbf e_{12} = -1$. The right contraction is $A \llcorner B = \langle A B \rangle_{j - k}$ for $j \geq k$.

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

### Identities

With $x$ a vector and $A$ of grade $j$:

$$
\begin{aligned}
(A \wedge B) \lrcorner C &= A \lrcorner (B \lrcorner C) \\
x \lrcorner (A \wedge B) &= (x \lrcorner A) \wedge B + (-1)^j A \wedge (x \lrcorner B) \\
x A &= x \lrcorner A + x \wedge A \\
\widetilde{A \lrcorner B} &= \tilde B \llcorner \tilde A \\
\tilde A \lrcorner A &= \text{form}(A)
\end{aligned}
$$

### Orthogonality

For orthogonal basis vectors $\mathbf{e}_m \perp \mathbf{e}_n$ with $m \neq n$:

```math
\mathbf{e}_m \lrcorner (\mathbf{e}_n \wedge \mathbf{e}_p) = 0 \quad \text{when } m \notin \{n, p\}
```

## Complement Operations

Complement operations map between dual grades using the Levi-Civita symbol.

### Right Complement

For grade-$k$ blade in $d$ dimensions:

$$
\overline{\mathbf{B}}^{m_{k + 1} \ldots m_d} = \frac{1}{k!} B^{m_1 \ldots m_k} ε_{m_1 \ldots m_d}
$$

### Left Complement

$$
\underline{\mathbf{B}}^{m_1 \ldots m_{d - k}} = \frac{1}{k!} ε_{m_1 \ldots m_d} B^{m_{d - k + 1} \ldots m_d}
$$

### Grade Mapping

```math
\text{grade}(\overline{\mathbf{B}}) = d - \text{grade}(\mathbf{B})
```

### Double Complement

$$
\begin{aligned}
\overline{\overline{\mathbf B}} &= (-1)^{k(d - k)} \mathbf B \\
\underline{\overline{\mathbf B}} &= \mathbf B
\end{aligned}
$$

### Volume and Norm

For basis blades, with $\mathbb 1$ the unit pseudoscalar:

$$
\begin{aligned}
\mathbf B \wedge \overline{\mathbf B} &= \mathbb 1 \\
\underline{\mathbf B} \wedge \mathbf B &= \mathbb 1
\end{aligned}
$$

In Euclidean space a complement has the norm of its element. Complements do not depend on the metric.

## Hodge Dual

The Hodge dual incorporates metric structure:

```math
(\star \mathbf{B})^{m_{k + 1} \ldots m_d} = \frac{1}{k!} B^{n_1 \ldots n_k} g_{n_1 m_1} \cdots g_{n_k m_k} ε^{m_1 \ldots m_d}
```

### Grade Mapping

```math
\text{grade}(\star \mathbf{B}) = d - \text{grade}(\mathbf{B})
```

### Identities

$$
\begin{aligned}
\mathbf A \wedge \star \mathbf B &= (\mathbf A \cdot \mathbf B) \mathbb 1 \\
\star \mathbf B &= \tilde{\mathbf B} \mathbb 1 \\
\star \star \mathbf B &= (-1)^{k(d - k)} \det(g) \mathbf B
\end{aligned}
$$

### Examples

Scalar dual:

```math
\star s = s \mathbb{1}
```

where $\mathbb{1}$ is the pseudoscalar.

3D vector dual:

$$
\star \mathbf{v} = \frac{1}{2} v^m ε^{mnp} \mathbf{e}_{np}
$$

3D bivector dual:

$$
\star (\mathbf{e}_1 \wedge \mathbf{e}_2) = \mathbf{e}_3
$$

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

For phasors $\mathbf{v} = \mathbf{A} e^{iφ}$, this gives the RMS amplitude $|\mathbf{A}|$.

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
\mathbf{e}_m \cdot \mathbf{e}_n = g_{mn} = δ_{mn}
```

## Projections

### Projection Formula

$$
\text{proj}_{\mathbf{B}}(\mathbf{A}) = (\mathbf{A} \lrcorner \mathbf{B}) \lrcorner \mathbf{B}^{-1}
$$

### Rejection Formula

```math
\text{rej}_{\mathbf{B}}(\mathbf{A}) = \mathbf{A} - \text{proj}_{\mathbf{B}}(\mathbf{A})
```

### Decomposition

```math
\mathbf{A} = \text{proj}_{\mathbf{B}}(\mathbf{A}) + \text{rej}_{\mathbf{B}}(\mathbf{A})
```

### Properties

- a blade that is not null projects onto itself unchanged, in any signature
- projecting twice changes nothing, and the rejection projects to zero
- the projection $\mathbf p$ of a vector satisfies $\mathbf p \wedge \mathbf B = 0$, and its rejection $\mathbf r$ satisfies $\mathbf r \lrcorner \mathbf B = 0$
- projection is an outermorphism, and does not depend on the scale or sign of $\mathbf B$
- a higher grade projects onto a lower one as zero, and so does anything onto a null blade

## Factorization

A blade factors into vectors that are orthogonal in the metric and wedge to it exactly, sign included, in any signature:

$$
\mathbf{B} = \mathbf{v}_1 \wedge \cdots \wedge \mathbf{v}_k
$$

A k-vector that is not a blade, such as $\mathbf e_{12} + \mathbf e_{34}$, raises an error.

## Join and Meet

### Join (Union)

$$
\text{join}(\mathbf{A}, \mathbf{B}) = \mathbf{A} \wedge \mathbf{B}
$$

### Meet (Intersection)

$$
\mathbf{A} \vee \mathbf{B} = \underline{\left(\overline{\mathbf{A}} \wedge \overline{\mathbf{B}}\right)}
$$

### Grade Behavior

Join:

$$
\text{grade}(\mathbf{A} \wedge \mathbf{B}) = j + k
$$

Meet (transverse):

$$
\text{grade}(\mathbf{A} \vee \mathbf{B}) = j + k - d
$$

### Meet Identities

$$
\begin{aligned}
\overline{\mathbf A \vee \mathbf B} &= \overline{\mathbf A} \wedge \overline{\mathbf B} \\
\mathbf A \vee \mathbb 1 &= \mathbf A \\
\mathbf A \vee \mathbf B &= (-1)^{(d - j)(d - k)} \mathbf B \vee \mathbf A
\end{aligned}
$$

The meet is associative and metric-independent, lies in both blades, and is scalar zero when $j + k \lt d$. Values: $\mathbf e_{12} \vee \mathbf e_{13} = \mathbf e_1$ and $\mathbf e_{12} \vee \mathbf e_{23} = \mathbf e_2$ in 3D.

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

## Complex Structure

In a space with modes the structure bivector and its map are:

```math
\begin{aligned}
\mathbf{J} &= \mathbf{e}_m \wedge \mathbf{e}_{\dot m} \\
J(\mathbf{v}) &= \mathbf{v} \lrcorner \mathbf{J}
\end{aligned}
```

### Structure

- $J \mathbf{e}_m = \mathbf{e}_{\dot m}$ and $J \mathbf{e}_{\dot m} = -\mathbf{e}_m$
- $J^2 = (-1)^k$ on grade $k$
- $J \mathbf{e}_{12} = \mathbf{e}_{\dot 1 \dot 2}$
- $\mathbf{J}^{\wedge n} / n! = \mathbb{1}$

### Blade Classes

- complex: $J \mathbf{b} = \mathbf{b}$, true for $\mathbf{e}_{1 \dot 1}$, for $\mathbf{v} \wedge J \mathbf{v}$, and for $\mathbb{1}$; never for odd grade
- isotropic: $\mathbf{J} \lrcorner \mathbf{b} = 0$, true for every vector
- Lagrangian: isotropic of grade $n$, such as $\mathbf{e}_{12}$, with $\mathbf{e}_{12} \wedge J \mathbf{e}_{12} = -\mathbb{1}$

### Kähler Angle

For a unit 2-blade, $\cos θ = \mathbf{b} \cdot \mathbf{J}$:

- $θ(\mathbf{e}_{1 \dot 1}) = 0$, $θ(\mathbf{e}_{\dot 1 1}) = π$, $θ(\mathbf{e}_{12}) = π / 2$
- $θ(\mathbf{e}_1 \wedge (\cos α \mathbf{e}_{\dot 1} + \sin α \mathbf{e}_2)) = α$
- a tangent frame and its blade give the same angle

### Amplitudes

- $z^m = v^m + i v^{\dot m}$, with no normalization factor
- multiplication by $i$ is $J$: $(J \mathbf{v})$ has amplitudes $i z$
