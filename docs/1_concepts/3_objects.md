# Objects in Geometric Algebra

This document describes the fundamental objects in geometric algebra: vectors, blades, multivectors, versors, and rotors.

## Terminology: What We Mean by "Vector"

In morphis, the term **Vector** refers to a homogeneous multivector of pure grade $k$. This is what other texts might call a "$k$-vector" or "$p$-vector". We adopt this naming because:

1. **Mathematical consistency**: A grade-$k$ element is a vector in the vector space $\bigwedge^k V$
2. **Simplicity**: Avoids the variable naming conventions ($k$-vector, $p$-vector) that differ between texts
3. **Clarity**: The grade is an explicit attribute, not embedded in the name

A `Vector` with `grade=1` is what's traditionally called a "vector". A `Vector` with `grade=2` is a bivector, etc.

```python
from morphis.elements import Vector, euclidean_metric

g = euclidean_metric(3)

# Grade-1 vector (traditional "vector")
v = Vector([1, 0, 0], grade=1, metric=g)

# Grade-2 vector (bivector)
b = Vector(data, grade=2, metric=g)

# Grade-0 vector (scalar)
s = Vector(1.5, grade=0, metric=g)
```

## Vectors ($k$-Vectors)

A **Vector** (homogeneous multivector) is an element of pure grade $k$. In components:

$$\mathbf{A}_k = A^{m_1 \ldots m_k} \mathbf{e}_{m_1 \ldots m_k}$$

where the basis $k$-vectors satisfy:

$$\mathbf{e}_{m_1 \ldots m_k} = \mathbf{e}_{m_1} \wedge \cdots \wedge \mathbf{e}_{m_k}$$

Properties of Vectors:
- **Fixed grade**: All components have the same grade
- **Antisymmetric**: $A^{\ldots m \ldots n \ldots} = -A^{\ldots n \ldots m \ldots}$
- **Form a vector space**: Can add, subtract, scalar multiply

```python
from morphis.elements import Vector, basis_vectors, euclidean_metric

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Bivector via wedge product
b = e1 ^ e2

# Check grade
b.grade  # 2

# Vectors of same grade can be added
c = e2 ^ e3
d = b + c  # Still grade-2
```

## Blades

A **blade** (or simple $k$-vector) is a Vector that can be written as the wedge product of $k$ grade-1 vectors:

$$\mathbf{B} = \mathbf{v}_1 \wedge \mathbf{v}_2 \wedge \cdots \wedge \mathbf{v}_k$$

Blades represent **oriented $k$-dimensional subspaces**. The magnitude encodes the $k$-dimensional volume, and the orientation determines the "sense" of the subspace.

Not every Vector is a blade. For example, in 4D:

$$\mathbf{e}_{12} + \mathbf{e}_{34}$$

is a bivector (grade-2 Vector) but cannot be factored as $\mathbf{a} \wedge \mathbf{b}$ for any grade-1 vectors $\mathbf{a}, \mathbf{b}$.

```python
from morphis.elements import Vector, basis_vectors, euclidean_metric

g = euclidean_metric(4)
e1, e2, e3, e4 = basis_vectors(g)

# This is a blade
b1 = e1 ^ e2
b1.is_blade  # True

# This is NOT a blade in 4D
b2 = (e1 ^ e2) + (e3 ^ e4)
b2.is_blade  # False
```

## Factorization

For blades, the `.span()` method returns a factorization into grade-1 vectors:

```python
from morphis.elements import basis_vectors, euclidean_metric

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

b = e1 ^ e2
vectors = b.span()  # Returns (v1, v2) where b = v1 ^ v2
```

Note: Factorization is not unique. Any $k$ linearly independent vectors spanning the same subspace yield the same blade (up to scalar).

## MultiVectors

A **MultiVector** is a general element of the Clifford algebra—a sum of Vectors of different grades:

$$\mathbf{M} = \sum_{k=0}^{d} \langle \mathbf{M} \rangle_k$$

where $\langle \mathbf{M} \rangle_k$ denotes the grade-$k$ projection.

In morphis, MultiVectors are stored sparsely as a dictionary mapping grade to Vector:

```python
from morphis.elements import MultiVector, Vector, euclidean_metric

g = euclidean_metric(3)
s = Vector(1.0, grade=0, metric=g)
v = Vector([1, 0, 0], grade=1, metric=g)
b = Vector(data, grade=2, metric=g)

# Create from Vectors
M = MultiVector(s, v, b)

# Grade selection
M.grades    # [0, 1, 2]
M[0]        # scalar component (Vector)
M[1]        # vector component (Vector)
M[2]        # bivector component (Vector)
M[3]        # None (not present)
```

## Versors

A **versor** is a product of invertible grade-1 vectors using the geometric product:

$$\mathbf{V} = \mathbf{v}_1 \mathbf{v}_2 \cdots \mathbf{v}_k$$

Versors generate orthogonal transformations via the **sandwich product**:

$$\mathbf{x}' = \mathbf{V} \mathbf{x} \mathbf{V}^{-1}$$

Key properties:
- **Closed under multiplication**: Versor $\times$ Versor = Versor
- **Invertible**: $\mathbf{V}^{-1}$ exists
- **Orthogonal action**: The sandwich product preserves norms

## Rotors

A **rotor** is an even versor (product of an even number of vectors) satisfying:

$$R \in \text{Cl}^+(V, g), \quad R \tilde{R} = \mathbf{1}$$

Rotors generate **rotations** (proper orthogonal transformations) via the sandwich product:

$$\mathbf{v}' = R \mathbf{v} \tilde{R}$$

The normalization $R \tilde{R} = 1$ ensures the transformation preserves norms.

```python
from morphis.elements import basis_vectors, euclidean_metric
from morphis.transforms import rotor
from numpy import pi

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Bivector defines rotation plane
b = e1 ^ e2

# Create rotor for 90-degree rotation
R = rotor(b.normalize(), pi/2)

# Check rotor properties
R.is_even   # True (only grades 0, 2)
R.is_rotor  # True (R * ~R = 1)

# Apply rotation
v = e1
v_rotated = R * v * ~R  # v becomes e2
```

## Motors (PGA)

In **Projective Geometric Algebra** (PGA), a **motor** combines rotation and translation in a single element. Motors have grades $\{0, 2\}$ and satisfy $M \tilde{M} = 1$.

$$M = R + \frac{\epsilon}{2} \mathbf{t} R$$

where $R$ is a rotor, $\mathbf{t}$ is the translation vector, and $\epsilon$ is the degenerate direction.

```python
from morphis.elements import pga_metric
from morphis.transforms import rotor, translator
from morphis.operations import geometric

h = pga_metric(3)  # 3D PGA

# Create motor from rotor and translator
R = rotor(b, angle)
T = translator(direction)
M = geometric(T, R)  # Motor = translate after rotate

M.is_motor  # True
```

## Summary Table

| Object | Definition | Properties |
|--------|-----------|------------|
| **Vector** | Pure grade $k$ element | Homogeneous, antisymmetric |
| **Blade** | Factorizable Vector | Represents $k$-dimensional subspace |
| **MultiVector** | Sum of Vectors | General Clifford algebra element |
| **Versor** | Product of invertible vectors | Generates orthogonal transformations |
| **Rotor** | Even unit versor | Generates rotations |
| **Motor** | PGA versor | Combines rotation and translation |
