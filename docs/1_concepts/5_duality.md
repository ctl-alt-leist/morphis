# Complements and Duality

Duality operations map between vectors of complementary grades, revealing the deep symmetry between intrinsic and extrinsic descriptions of geometric objects.

## The Unit Pseudoscalar

In any $d$-dimensional space, the **unit pseudoscalar** serves as the fundamental reference element:

$$\mathbb{1} = \mathbf{e}_1 \wedge \mathbf{e}_2 \wedge \cdots \wedge \mathbf{e}_d$$

This highest-grade element represents the oriented volume of the entire space and provides the algebraic foundation for all duality operations.

```python
from morphis.elements import pseudoscalar, euclidean_metric

g = euclidean_metric(3)
I = pseudoscalar(g)  # e1 ^ e2 ^ e3
I.grade  # 3 (equals dim)
```

## Complement Operations

Complements map between grades $k$ and $(d - k)$ using only the **Levi-Civita symbol**—they are metric-independent.

### Right Complement

For a grade-$k$ k-vector $b$:

$$
\bar{b}^{m_{k + 1} \ldots m_d} = b^{m_1 \ldots m_k} \, \varepsilon_{m_1 \ldots m_d}
$$

### Left Complement

$$
\underline{b}^{m_1 \ldots m_{d - k}} = \varepsilon_{m_1 \ldots m_d} \, b^{m_{d - k + 1} \ldots m_d}
$$

### Orthogonality

Complements satisfy the fundamental orthogonality relationship:

$$
u \wedge \overline{u} = \mathbb{1}
$$

$$
\underline{u} \wedge u = \mathbb{1}
$$

This reveals that a k-vector and its complement span the entire space with no overlap.

### Sign Relationship

$$
\underline{u} = (-1)^{\text{grade}(u) \cdot \text{antigrade}(u)} \, \overline{u}
$$

where antigrade = $d - $ grade.

### Involution Property

Complements are involutions—applying them twice returns the original:

$$
\overline{\overline{u}} = u
$$

$$
\underline{\underline{u}} = u
$$

```python
from morphis.operations import right_complement, left_complement

b_right = right_complement(b)
b_left = left_complement(b)
```

## Geometric Interpretation

Consider a plane in 3D space. We can describe it:

**Intrinsically**: Two vectors spanning it $\rightarrow$ bivector $\mathbf{p} = \mathbf{a} \wedge \mathbf{b}$

**Extrinsically**: The perpendicular vector $\rightarrow$ its complement $\bar{\mathbf{p}}$

The complement transforms between these descriptions. The relationship $\mathbf{p} \wedge \bar{\mathbf{p}} = \mathbb{1}$ ensures that the plane and its normal together span the full 3D space.

This generalizes to higher dimensions:
- In 4D: A plane (bivector) has a 2D complement (also a bivector)
- In 5D: A plane has a 3D complement (trivector)

## The Hodge Dual

The **Hodge dual** is the metric-dependent counterpart to complements:

$$
\star v = G(\bar{v})
$$

where $G$ applies the metric to raise/lower indices.

In components:

$$
(\star v)^{m_{k + 1} \ldots m_d} = \frac{1}{k!} \, g^{m_{k + 1} n_{k + 1}} \cdots g^{m_d n_d} \, v^{m_1 \ldots m_k} \, \varepsilon_{m_1 \ldots m_d}
$$

The key distinction:
- **Complement**: Uses only Levi-Civita symbol (metric-independent)
- **Hodge dual**: Uses Levi-Civita AND metric tensor (metric-dependent)

```python
from morphis.operations import hodge_dual

# or use the method
b_dual = b.hodge()

# Function syntax
b_dual = hodge_dual(b)
```

### Grade Mapping

$$
\text{grade}(\star b) = d - \text{grade}(b)
$$

### Double Hodge Dual

$$
\star \star v = (-1)^{k(d - k)} \, \text{sgn}(g) \, v
$$

where $\text{sgn}(g)$ is the sign of the metric determinant:
- Euclidean: $\text{sgn}(g) = +1$
- Lorentzian: $\text{sgn}(g) = -1$

### Examples by Dimension

**3D Euclidean:**
- Vector $v \xrightarrow{\star}$ bivector (perpendicular plane)
- Bivector $b \xrightarrow{\star}$ vector (perpendicular direction)
- Trivector $\xrightarrow{\star}$ scalar

**4D Euclidean:**
- Vector $\xrightarrow{\star}$ trivector
- Bivector $\xrightarrow{\star}$ bivector (self-dual structure!)
- Trivector $\xrightarrow{\star}$ vector

The self-dual nature of bivectors in 4D creates rich structure—the 6-dimensional bivector space splits into 3D self-dual and anti-self-dual subspaces.

## Metric-Independent vs Metric-Dependent

Understanding which operations require the metric is essential:

**Metric-independent (work in any vector space):**
- Exterior product (wedge) $\wedge$
- Left and right complements
- Join operation (same as wedge)
- Meet operation (via complements)

**Metric-dependent (require inner product structure):**
- Interior product (contraction)
- Hodge dual
- Norms and normalization
- Orthogonal projection and rejection
- Distance calculations

```python
# Metric-independent: only needs dimension
b_complement = right_complement(b)

# Metric-dependent: uses the metric tensor
b_dual = hodge_dual(b)  # Requires metric
```

## The Meet via Complements

The intersection (meet) of two subspaces can be computed using complements:

$$
u \vee v = \overline{\left(\overline{u} \wedge \overline{v}\right)}
$$

This duality formula converts intersection to a wedge product in the complement space.

```python
from morphis.operations import meet

# Intersection of two subspaces
intersection = meet(u, v)
```

## Applications

### Cross Product as Duality

The 3D cross product is secretly the Hodge dual of the wedge product:

$$\mathbf{a} \times \mathbf{b} = \star(\mathbf{a} \wedge \mathbf{b})$$

This explains why the cross product only works in 3D—it requires the special coincidence that vectors and bivectors have the same dimension (both 3).

### Electromagnetic Duality

In electromagnetism, the electric field $\mathbf{E}$ and magnetic field $\mathbf{B}$ are Hodge duals in spacetime. The Maxwell equations exhibit a duality symmetry under $\mathbf{E} \leftrightarrow \star\mathbf{B}$.

### Computational Efficiency

Duality operations enable computing in whichever grade is more convenient:

- Finding intersection of $(d - 1)$-dimensional hyperplanes: work with their 1-dimensional normal vectors
- Testing incidence: work in the simpler dual space

The involution property $\overline{\overline{u}} = u$ ensures all such transformations are reversible.
