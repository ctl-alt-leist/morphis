# Transformations

Geometric algebra provides a unified framework for all orthogonal transformations: reflections, rotations, translations, and their compositions. The key is the **sandwich product**.

## The Sandwich Product

Transformations in geometric algebra act via:

$$\mathbf{x}' = M \mathbf{x} \tilde{M}$$

or for reflections:

$$\mathbf{x}' = -\mathbf{n} \mathbf{x} \mathbf{n}$$

This pattern:
- Preserves geometric structure (grade, norm)
- Is an outermorphism (preserves wedge products)
- Composes naturally: $(M_2 M_1) \mathbf{x} \widetilde{(M_2 M_1)} = M_2 (M_1 \mathbf{x} \tilde{M}_1) \tilde{M}_2$

```python
from morphis.transforms import transform

# General transformation
x_transformed = transform(x, M)  # M * x * ~M
```

## Reflections

A reflection through the hyperplane perpendicular to unit vector $\mathbf{n}$:

$$\mathbf{v}' = -\mathbf{n} \mathbf{v} \mathbf{n}$$

Properties:
- **Involution**: Two reflections give identity
- **Magnitude preserving**: $|\mathbf{v}'| = |\mathbf{v}|$
- **Determinant**: $\det = -1$ (orientation reversing)

### Cartan-Dieudonné Theorem

Any orthogonal transformation in $d$ dimensions factors as at most $d$ reflections.

## Rotors

A **rotor** is an even versor satisfying $R \tilde{R} = 1$. It represents rotation:

$$\mathbf{v}' = R \mathbf{v} \tilde{R}$$

### Construction from Bivector

$$R = e^{-\mathbf{B}\theta/2} = \cos(\theta/2) - \sin(\theta/2)\hat{\mathbf{B}}$$

where $\hat{\mathbf{B}}$ is the unit bivector defining the rotation plane.

```python
from morphis.transforms import rotor
from morphis.elements import euclidean_metric, basis_vectors
from numpy import pi

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Rotation by 90 degrees in the xy-plane
b = (e1 ^ e2).normalize()
R = rotor(b, pi/2)

# Apply
v_rotated = R * e1 * ~R  # e1 -> e2
```

### Rodrigues Formula

The rotation action expands to:

$$\mathbf{x}' = \mathbf{x} + \sin(\theta) \, \mathbf{B} \cdot \mathbf{x} + (1 - \cos(\theta)) \, \mathbf{B} \cdot (\mathbf{B} \cdot \mathbf{x})$$

### Two Reflections = One Rotation

A rotor equals the product of two reflection vectors:

$$R = \mathbf{n}_2 \mathbf{n}_1$$

The rotation angle is twice the angle between the reflection planes.

## Rotations in Higher Dimensions

In $d$ dimensions, there are $\binom{d}{2}$ independent rotation planes. In 4D:

$$\{\mathbf{e}_{12}, \mathbf{e}_{13}, \mathbf{e}_{14}, \mathbf{e}_{23}, \mathbf{e}_{24}, \mathbf{e}_{34}\}$$

A general 4D rotation may involve multiple simultaneous planes (double rotation).

```python
from morphis.elements import euclidean_metric, basis_vectors

g = euclidean_metric(4)
e1, e2, e3, e4 = basis_vectors(g)

# Rotation in two planes simultaneously
b1 = e1 ^ e2
b2 = e3 ^ e4
R = rotor(b1.normalize(), theta1) * rotor(b2.normalize(), theta2)
```

## Translators (PGA)

In Projective Geometric Algebra, translation is a versor transformation.

### Construction

For translation by vector $\mathbf{t}$:

$$T = e^{-\mathbf{t}_0/2} = 1 - \frac{1}{2}t^m \mathbf{e}_{0m}$$

where $\mathbf{e}_{0m}$ are degenerate bivectors (involving the ideal direction $\mathbf{e}_0$).

```python
from morphis.transforms import translator, direction
from morphis.elements import pga_metric

h = pga_metric(3)

# Translation vector
d = direction([1, 0, 0], metric=h)
T = translator(d)

# Apply translation
p_translated = T * p * ~T
```

### Properties

- **Commutative**: $T_{\mathbf{s}} T_{\mathbf{t}} = T_{\mathbf{t}} T_{\mathbf{s}}$
- **Additive composition**: $T_{\mathbf{s}} T_{\mathbf{t}} = T_{\mathbf{s} + \mathbf{t}}$
- **Nilpotent generator**: $\mathbf{t}^2 = 0$

## Motors

A **motor** combines rotation and translation in PGA:

$$M = RT$$

or via line exponential:

$$M = e^{-L\theta/2}$$

where $L$ is a line (bivector in PGA).

```python
from morphis.transforms import rotation_about_point

# Rotation about a point (not the origin)
M = rotation_about_point(center, b, angle)

# Motor via composition
M = T * R  # Translate after rotate
M = R * T  # Rotate after translate (different result!)
```

### Non-Commutativity

Rotation and translation do not commute:

$$RT \neq TR$$

"Rotate then translate" differs from "translate then rotate."

## Projective Geometric Algebra (PGA)

PGA embeds $d$-dimensional Euclidean space in a $(d+1)$-dimensional Clifford algebra with degenerate metric.

### Metric Structure

$$g = \text{diag}(0, 1, 1, \ldots, 1)$$

The $\mathbf{e}_0$ direction is **ideal** (at infinity).

### Point Representation

A Euclidean point $\mathbf{x} = (x^1, \ldots, x^d)$ embeds as:

$$\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$$

```python
from morphis.transforms import point, euclidean
from morphis.elements import pga_metric

h = pga_metric(3)

# Create a point
p = point([1, 2, 3], metric=h)

# Extract Euclidean coordinates
coords = euclidean(p)  # [1, 2, 3]
```

### Direction Representation

A direction (point at infinity):

$$\mathbf{d} = v^m \mathbf{e}_m$$

No $\mathbf{e}_0$ component.

```python
from morphis.transforms import direction, is_direction

d = direction([1, 0, 0], metric=h)
is_direction(d)  # True
```

### Geometric Constructors

**Line through two points**:
$$\ell = \mathbf{p} \wedge \mathbf{q}$$

**Plane through three points**:
$$\pi = \mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r}$$

```python
from morphis.transforms import line, plane

L = line(p1, p2)
P = plane(p1, p2, p3)
```

### Incidence Predicates

**Collinearity** (points on a line):
$$\mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r} = 0$$

**Point on line**:
$$\mathbf{p} \wedge \ell = 0$$

```python
from morphis.transforms import are_collinear, point_on_line

are_collinear(p, q, r)
point_on_line(p, L)
```

### Distance Functions

**Point to point**:
$$d(\mathbf{p}, \mathbf{q}) = |\mathbf{p} \wedge \mathbf{q}| / (w(\mathbf{p}) \cdot w(\mathbf{q}))$$

**Point to line**:
$$d(\mathbf{p}, \ell) = |\mathbf{p} \wedge \ell| / |\ell|$$

```python
from morphis.transforms import distance_point_to_point, distance_point_to_line

d = distance_point_to_point(p1, p2)
d = distance_point_to_line(p, L)
```

## Geometric Invariants

All motor transformations preserve:

- **Distance**: $d(\mathbf{p}'_1, \mathbf{p}'_2) = d(\mathbf{p}_1, \mathbf{p}_2)$
- **Angle**: $\cos\theta' = \cos\theta$
- **Orientation**: $\det(M) = +1$

These are the defining properties of rigid motions (Euclidean isometries).

## Summary Table

| Transform | Generator | Versor | Action |
|-----------|-----------|--------|--------|
| Reflection | Unit vector $\mathbf{n}$ | Odd | $-\mathbf{n}\mathbf{x}\mathbf{n}$ |
| Rotation | Bivector $\mathbf{B}$ | Even (rotor) | $R\mathbf{x}\tilde{R}$ |
| Translation | Degenerate bivector | Even (translator) | $T\mathbf{x}\tilde{T}$ |
| Rigid motion | Line $L$ | Even (motor) | $M\mathbf{x}\tilde{M}$ |
| Boost | Timelike bivector | Even | $B\mathbf{x}\tilde{B}$ |
