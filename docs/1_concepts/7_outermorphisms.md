# Outermorphisms

An **outermorphism** (or exomorphism) is a linear map between exterior algebras that preserves the wedge product. This preservation property means that outermorphisms are completely determined by their action on grade-1 vectors.

## Definition

An outermorphism $f: \bigwedge V \to \bigwedge W$ preserves the wedge product:

$$f(\mathbf{a} \wedge \mathbf{b}) = f(\mathbf{a}) \wedge f(\mathbf{b})$$

This has a profound consequence: if $A: V \to W$ is a linear map on grade-1 vectors, it extends **uniquely** to an outermorphism on all grades.

## The Exterior Power

Given a linear map $A: V \to W$ on vectors, its **$k$-th exterior power** $\bigwedge^k A$ acts on grade-$k$ vectors:

$$(\bigwedge^k A)(\mathbf{v}_1 \wedge \mathbf{v}_2 \wedge \cdots \wedge \mathbf{v}_k) = A(\mathbf{v}_1) \wedge A(\mathbf{v}_2) \wedge \cdots \wedge A(\mathbf{v}_k)$$

### Component Form

For a $d \times d$ matrix $A^i{}_j$ and grade-$k$ vector $B^{m_1 \ldots m_k}$:

$$(\bigwedge^k A)(\mathbf{B})^{i_1 \ldots i_k} = A^{i_1}{}_{m_1} \cdots A^{i_k}{}_{m_k} \, B^{m_1 \ldots m_k}$$

This is simply $k$ copies of $A$ contracting with the $k$ indices of the vector—a natural einsum operation.

### Special Cases

**Scalars (grade 0)** are unchanged:
$$(\bigwedge^0 A)(s) = s$$

**Pseudoscalar (grade $d$)** scales by determinant:
$$(\bigwedge^d A)(\mathbb{1}) = \det(A) \cdot \mathbb{1}$$

This last property explains why the determinant represents volume scaling.

## In Morphis: The Operator Class

The `Operator` class represents linear maps. When an Operator maps grade-1 to grade-1, it can extend to act on all grades as an outermorphism.

```python
from morphis.elements import euclidean_metric
from morphis.operations import Operator
from morphis.algebra import VectorSpec
from numpy import array, cos, sin

g = euclidean_metric(3)

# A rotation matrix
theta = 0.5
R = array([
    [cos(theta), -sin(theta), 0],
    [sin(theta),  cos(theta), 0],
    [0,           0,          1],
])

# Create operator from matrix
L = Operator(
    data=R,
    input_spec=VectorSpec(grade=1, collection=0, dim=3),
    output_spec=VectorSpec(grade=1, collection=0, dim=3),
    metric=g,
)

# Check if it's an outermorphism
L.is_outermorphism  # True (grade-1 to grade-1)

# Extract the vector map
L.vector_map  # The 3x3 matrix
```

## Applying Outermorphisms

An outermorphism operator can apply to vectors of **any grade**:

```python
from morphis.elements import basis_vectors

e1, e2, e3 = basis_vectors(g)

# Apply to grade-1 vector
v = e1
v_transformed = L * v

# Apply to grade-2 vector (uses 2nd exterior power)
b = e1 ^ e2
b_transformed = L * b

# Apply to full MultiVector (each grade transformed)
N = L * M
```

The exterior power is computed on-demand: $k$ copies of the vector map contract with the $k$ indices of the input.

## Composition

Outermorphism composition corresponds to matrix multiplication:

$$(f \circ g)|_{\text{grade-}k} = \bigwedge^k(AB)$$

where $A$ and $B$ are the vector maps of $f$ and $g$.

```python
# Composition via multiplication
L_composed = L1 * L2  # L1(L2(x))
```

## Connection to Versors

The **sandwich product** by a versor $V$ defines an outermorphism:

$$\mathbf{x} \mapsto V \mathbf{x} V^{-1}$$

For a rotor $R$, this gives a rotation. The fact that $R \mathbf{B} \tilde{R}$ preserves grade for vectors is precisely because the sandwich product is an outermorphism when $V$ is a versor.

## Implementation Details

Morphis computes exterior powers efficiently:

1. Store only the $d \times d$ grade-1 map
2. Compute $\bigwedge^k A$ on demand via einsum
3. For grade-$k$: `einsum("ia,jb,...,...ab->...ij", A, A, ..., B)`

This approach:
- Uses $O(d^2)$ storage instead of $O(2^d \times 2^d)$
- Leverages existing einsum operations
- Maintains mathematical clarity

## Properties Summary

| Property | Statement |
|----------|-----------|
| Wedge preservation | $f(\mathbf{a} \wedge \mathbf{b}) = f(\mathbf{a}) \wedge f(\mathbf{b})$ |
| Determined by grade-1 | Know $A|_{V}$ $\Rightarrow$ know $f$ on all grades |
| Scalar invariance | $f(s) = s$ |
| Determinant property | $f(\mathbb{1}) = \det(A) \cdot \mathbb{1}$ |
| Composition | $(f \circ g)_k = \bigwedge^k(AB)$ |

## Non-Outermorphism Operators

Not all Operators are outermorphisms. For example:

```python
# Maps scalars to bivectors: NOT an outermorphism
G = Operator(
    data=G_data,
    input_spec=VectorSpec(grade=0, collection=1, dim=3),
    output_spec=VectorSpec(grade=2, collection=1, dim=3),
    metric=g,
)

G.is_outermorphism  # False (grade changes)
```

Such operators can still be applied to vectors matching their input spec, but cannot extend to arbitrary grades.
