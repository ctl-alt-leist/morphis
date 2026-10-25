# Transforms: Mathematical Properties

Mathematical foundations for rotors, translators, motors, and projective geometry operations.

## Rotors

A rotor represents rotation via the exponential map from bivector to transformation.

### Exponential Form

For unit bivector $B$ with $B^2 = -1$ and angle $\theta$:

```math
R = e^{-B\theta/2} = \cos(\theta/2) - \sin(\theta/2) B
```

### Multivector Structure

```math
R = R_0 \mathbf{1} + R^{mn} \mathbf{e}_{mn}
```

where:
- $R_0 = \cos(\theta/2)$
- $R^{mn} = -\sin(\theta/2) B^{mn}$

### Versor Property

```math
R \tilde{R} = 1
```

Proof:

```math
R \tilde{R} = \cos^2(\theta/2) + \sin^2(\theta/2) = 1
```

### Inverse

```math
R^{-1} = \tilde{R} = \cos(\theta/2) + \sin(\theta/2) B
```

### Identity

```math
R(0) = 1
```

## Sandwich Product

Rotors act via:

```math
\mathbf{p}' = R \mathbf{p} \tilde{R}
```

### Rodrigues Formula

For rotation in plane $B$ by angle $\theta$:

```math
x'^m = x^m + \sin(\theta) B^{mp} x_p + (1 - \cos(\theta)) B^{mp} B_p^{\ q} x_q
```

### Rotation in xy-Plane

With $B = \mathbf{e}_{12}$:

```math
\begin{align}
x' &= x \cos\theta - y \sin\theta \\
y' &= x \sin\theta + y \cos\theta \\
z' &= z
\end{align}
```

### Specific Rotations

90 degrees ($\theta = \pi/2$):

```math
(1, 0, 0) \mapsto (0, 1, 0)
```

180 degrees ($\theta = \pi$):

```math
(x, y, z) \mapsto (-x, -y, z)
```

### Plane Invariance

```math
R B \tilde{R} = B
```

## Translators

A translator represents pure translation in PGA.

### Exponential Form

```math
T = e^{-\mathbf{t}/2} = 1 - \frac{1}{2} t^m \mathbf{e}_{0m}
```

where $\mathbf{t} = t^m \mathbf{e}_{0m}$ is the degenerate bivector.

### Truncation

Degenerate bivectors square to zero:

```math
\mathbf{t}^2 = 0
```

so the exponential truncates to linear term.

### Translation Action

```math
\mathbf{p}' = T \mathbf{p} \tilde{T}
```

For point $\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$:

```math
\mathbf{p}' = \mathbf{e}_0 + (x^m + t^m) \mathbf{e}_m
```

### Composition

Translators compose additively:

```math
T_{\mathbf{t}} T_{\mathbf{s}} = T_{\mathbf{t} + \mathbf{s}}
```

### Commutativity

```math
T_{\mathbf{t}} T_{\mathbf{s}} = T_{\mathbf{s}} T_{\mathbf{t}}
```

### Versor Property

```math
T \tilde{T} = 1
```

## Motors

Motors combine rotation and translation.

### Composition

```math
M = RT
```

or via line exponential:

```math
M = e^{-L\theta/2}
```

### Rotation About Arbitrary Center

```math
M = T_{\mathbf{c}} R T_{-\mathbf{c}}
```

### Center Invariance

```math
M \mathbf{p}_c \tilde{M} = \mathbf{p}_c
```

### Non-Commutativity

```math
RT \neq TR
```

"Rotate then translate" differs from "translate then rotate."

### Associativity

```math
(M_3 M_2) M_1 = M_3 (M_2 M_1)
```

### Closure

- Rotor $\times$ Rotor = Rotor
- Translator $\times$ Translator = Translator
- Rotor $\times$ Translator = Motor

All maintain versor property: $M \tilde{M} = 1$.

## Geometric Invariants

### Distance Preservation

```math
d(\mathbf{p}'_1, \mathbf{p}'_2) = d(\mathbf{p}_1, \mathbf{p}_2)
```

### Angle Preservation

```math
\cos\theta' = \cos\theta
```

### Orientation Preservation

```math
\det(M) = +1
```

## Reflections

### Formula

```math
\mathbf{v}' = -\mathbf{n}\mathbf{v}\mathbf{n}
```

### Properties

Involution (twice = identity):

```math
-\mathbf{n}(-\mathbf{n}\mathbf{v}\mathbf{n})\mathbf{n} = \mathbf{v}
```

Magnitude preservation:

```math
|\mathbf{v}'| = |\mathbf{v}|
```

## Versors

### From Reflections

```math
V = \mathbf{n}_k \cdots \mathbf{n}_2 \mathbf{n}_1
```

### Two Reflections = Rotation

```math
\mathbf{n}_2 \mathbf{n}_1 = R_{2\theta}
```

where $\theta$ is angle between reflection planes.

### Cartan-Dieudonne

Any orthogonal transformation in $d$ dimensions factors as at most $d$ reflections.

## Projective Geometry (PGA)

### Point Embedding

```math
\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m
```

### Direction Embedding

```math
\mathbf{d} = v^m \mathbf{e}_m
```

(zero ideal component)

### Weight and Bulk

```math
w(\mathbf{p}) = p^0
```

```math
\text{bulk}(\mathbf{p}) = (p^1, \ldots, p^d)
```

### Euclidean Projection

```math
\text{euclidean}(\mathbf{p}) = \frac{\text{bulk}(\mathbf{p})}{w(\mathbf{p})}
```

### Geometric Constructors

Line through points:

```math
\ell = \mathbf{p} \wedge \mathbf{q}
```

Plane through points:

```math
\pi = \mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r}
```

### Distance Functions

Point-to-point:

```math
d(\mathbf{p}, \mathbf{q}) = \sqrt{g_{mn} (x_q^m - x_p^m)(x_q^n - x_p^n)}
```

Point-to-line:

```math
d(\mathbf{p}, \ell) = \frac{|\mathbf{p} \wedge \ell|}{|\ell|}
```

### Incidence Predicates

Collinearity:

```math
\mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r} = 0
```

Coplanarity:

```math
\mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r} \wedge \mathbf{s} = 0
```

Point on line:

```math
\mathbf{p} \wedge \ell = 0
```

## Higher Dimensions

### Rotation Planes

In $d$ dimensions: $\binom{d}{2}$ independent rotation planes.

4D bivector basis:

```math
\mathbf{e}_{12}, \mathbf{e}_{13}, \mathbf{e}_{14}, \mathbf{e}_{23}, \mathbf{e}_{24}, \mathbf{e}_{34}
```

### Component Counts

Motor components: $1 + \binom{d + 1}{2}$

- 3D: $1 + 6 = 7$
- 4D: $1 + 10 = 11$

### Half-Angle Convention

Exponential uses half-angles so sandwich product gives full rotation:

```math
R \mathbf{v} \tilde{R} \text{ rotates by } \theta
```
