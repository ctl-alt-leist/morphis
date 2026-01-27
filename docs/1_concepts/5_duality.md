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

Complements map between grades $k$ and $(d-k)$ using only the **Levi-Civita symbol**—they are metric-independent.

### Right Complement

For a grade-$k$ vector $\mathbf{B}$:

$$\bar{\mathbf{B}}^{m_{k+1} \ldots m_d} = B^{m_1 \ldots m_k} \, \varepsilon_{m_1 \ldots m_d}$$

### Left Complement

$$\underline{\mathbf{B}}^{m_1 \ldots m_{d-k}} = \varepsilon_{m_1 \ldots m_d} \, B^{m_{d-k+1} \ldots m_d}$$

### Orthogonality

Complements satisfy the fundamental orthogonality relationship:

$$\mathbf{u} \wedge \overline{\mathbf{u}} = \mathbb{1}$$

$$\underline{\mathbf{u}} \wedge \mathbf{u} = \mathbb{1}$$

This reveals that a vector and its complement span the entire space with no overlap.

### Sign Relationship

$$\underline{\mathbf{u}} = (-1)^{\text{grade}(\mathbf{u}) \cdot \text{antigrade}(\mathbf{u})} \, \overline{\mathbf{u}}$$

where antigrade = $d - $ grade.

### Involution Property

Complements are involutions—applying them twice returns the original:

$$\overline{\overline{\mathbf{u}}} = \mathbf{u}$$

$$\underline{\underline{\mathbf{u}}} = \mathbf{u}$$

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

$$\star \mathbf{V} = G(\bar{\mathbf{V}})$$

where $G$ applies the metric to raise/lower indices.

In components:

$$(\star V)^{m_{k+1} \ldots m_d} = \frac{1}{k!} g^{m_{k+1} n_{k+1}} \cdots g^{m_d n_d} \, V^{m_1 \ldots m_k} \, \varepsilon_{m_1 \ldots m_d}$$

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

$$\text{grade}(\star \mathbf{B}) = d - \text{grade}(\mathbf{B})$$

### Double Hodge Dual

$$\star \star \mathbf{V} = (-1)^{k(d-k)} \, \text{sgn}(g) \, \mathbf{V}$$

where $\text{sgn}(g)$ is the sign of the metric determinant:
- Euclidean: $\text{sgn}(g) = +1$
- Lorentzian: $\text{sgn}(g) = -1$

### Examples by Dimension

**3D Euclidean:**
- Vector $\mathbf{v} \xrightarrow{\star}$ bivector (perpendicular plane)
- Bivector $\mathbf{B} \xrightarrow{\star}$ vector (perpendicular direction)
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

$$\mathbf{A} \vee \mathbf{B} = \overline{\left(\overline{\mathbf{A}} \wedge \overline{\mathbf{B}}\right)}$$

This duality formula converts intersection to a wedge product in the complement space.

```python
from morphis.operations import meet

# Intersection of two subspaces
intersection = meet(A, B)
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

- Finding intersection of $(d-1)$-dimensional hyperplanes: work with their 1-dimensional normal vectors
- Testing incidence: work in the simpler dual space

The involution property $\overline{\overline{\mathbf{u}}} = \mathbf{u}$ ensures all such transformations are reversible.
