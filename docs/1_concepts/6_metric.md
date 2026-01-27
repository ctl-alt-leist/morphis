# The Metric

The metric introduces measurement into geometric algebra—lengths, angles, volumes, and the distinction between positional and directional information.

## The Metric Tensor

The **metric tensor** $g_{ab}$ emerges from the inner products of basis vectors:

$$g_{ab} = \mathbf{e}_a \cdot \mathbf{e}_b$$

This symmetric bilinear form encodes all geometric information about how vectors relate metrically:
- **Euclidean**: $g_{ab} = \delta_{ab}$ (identity matrix)
- **Minkowski**: signature $(+,-,-,-)$ or $(-,+,+,+)$
- **Degenerate (PGA)**: some diagonal entries are zero

```python
from morphis.elements import euclidean_metric, pga_metric, lorentzian_metric

# Euclidean 3D: diag(1, 1, 1)
g = euclidean_metric(3)

# PGA 3D: diag(0, 1, 1, 1) in 4D ambient space
# Basis: e0 (ideal direction), e1, e2, e3
h = pga_metric(3)

# Minkowski 4D: diag(-1, 1, 1, 1)
# Basis: e0 (time), e1, e2, e3
eta = lorentzian_metric(4)

# Access properties
g.dim        # 3
g.tensor     # The metric matrix
g.signature  # (3, 0, 0)  (positive, negative, zero)
```

## Extended Inner Products

The metric extends from vectors to all grades. For grade-$k$ k-vectors:

$$
(u_k \cdot v_k) = \frac{1}{k!} \, u^{m_1 \ldots m_k} v^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}
$$

For bivectors specifically:

$$
\mathbf{e}_{ij} \cdot \mathbf{e}_{kl} = g_{ik}g_{jl} - g_{il}g_{jk}
                                    = \begin{vmatrix} g_{ik} & g_{il} \\ g_{jk} & g_{jl} \end{vmatrix}
$$

The determinant structure generalizes to all grades:

$$
(u_1 \wedge \cdots \wedge u_k) \cdot (v_1 \wedge \cdots \wedge v_k) = \det(u_i \cdot v_j)
$$

## Norms

### Squared Norm (Bilinear)

For a grade-$k$ k-vector $b$:

$$
|b|^2 = \frac{1}{k!} \, b^{m_1 \ldots m_k} b^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}
$$

The factorial prevents overcounting due to antisymmetry.

```python
from morphis.operations import norm_squared, norm, normalize

# Squared norm (can be negative in non-Euclidean signatures)
ns = norm_squared(b)

# Norm (absolute value of squared norm, then sqrt)
n = norm(b)

# Normalized vector
b_unit = normalize(b)
# or
b_unit = b.normalize()
```

### Norm Properties

For Euclidean metrics:
- $|\mathbf{e}_m|^2 = g_{mm} = 1$
- $|v|^2 = \sum_m (v^m)^2$ (Pythagorean)

For non-Euclidean metrics:
- Squared norm can be negative (spacelike/timelike in Minkowski)
- Null vectors have $|v|^2 = 0$

### Hermitian Norm (for Phasors)

For complex (phasor) k-vectors, the **Hermitian norm** uses complex conjugation:

$$
|b|^2_H = \frac{1}{k!} \, \overline{b^{m_1 \ldots m_k}} \, b^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}
$$

This always returns real values for real metrics.

```python
from morphis.operations import hermitian_norm_squared, hermitian_norm

# For phasor fields
mag_squared = hermitian_norm_squared(b)
mag = hermitian_norm(b)
```

### Normalization and Zero Vectors

$$
\hat{b} = \frac{b}{|b|}
$$

Zero k-vectors normalize to zero (handled safely):

$$
\text{normalize}(0) = 0
$$

## Bulk and Weight (PGA)

In Projective Geometric Algebra, the metric enables decomposition into **bulk** (positional) and **weight** (directional) components.

### Definitions

For multivector $\mathbf{u}$:
- **Bulk**: $\mathbf{u}_{\bullet} = G\mathbf{u}$ (positional information)
- **Weight**: $\mathbf{u}_{\circ} = \mathbb{G}\mathbf{u}$ (directional information)

where $G$ is the metric exomorphism and $\mathbb{G}$ is the anti-metric exomorphism.

### Geometric Interpretation

- $\mathbf{u}_{\bullet} = 0$: The object contains the origin
- $\mathbf{u}_{\circ} = 0$: The object is at infinity (horizon)

### Weight Product Vanishing

$$\mathbf{a}_{\circ} \wedge \mathbf{b}_{\circ} = 0$$

Weight components, being purely directional, cannot span higher-dimensional spaces by themselves.

```python
from morphis.transforms import bulk, weight

# Decompose a PGA element
b = bulk(p)   # Positional part
w = weight(p) # Directional part
```

## Metric as Bridge

The metric connects $V$ to $V^*$ via index raising and lowering:

**Lowering** (musical flat $\flat$):
$$\mathbf{v}^\flat = g(\mathbf{v}, \cdot) \in V^*, \quad v_m = g_{mn} v^n$$

**Raising** (musical sharp $\sharp$):
$$\omega^\sharp = g^{-1}(\omega, \cdot) \in V, \quad \omega^m = g^{mn} \omega_n$$

where $g^{mn}$ is the inverse metric satisfying $g^{mp} g_{pn} = \delta^m_n$.

## Metric-Dependent Operations

These operations require the metric:

| Operation | Description |
|-----------|-------------|
| Interior product | Contraction using metric |
| Hodge dual | Complement + metric |
| Norm | Length measurement |
| Projection | Orthogonal decomposition |
| Distance | Metric distance between elements |
| Inverse | Uses norm for computation |

## The Exomorphism

The metric induces an **exomorphism** $G$ that extends metric relationships from vectors to all grades while preserving exterior product structure:

$$G(\mathbf{a} \wedge \mathbf{b}) = G(\mathbf{a}) \wedge G(\mathbf{b})$$

The metric exomorphism satisfies:

$$GG = \det(g) \, \mathbf{I}$$

This relationship ensures invertibility (when the metric is non-degenerate).

## Summary

The metric is the single piece of additional structure that elevates bare linear algebra to geometry. It:
- Defines inner products and lengths
- Connects $V$ to $V^*$
- Determines which rotations preserve it
- Specifies the Clifford relation $\mathbf{v}^2 = g(\mathbf{v}, \mathbf{v})$
- Enables measurement of geometric quantities
