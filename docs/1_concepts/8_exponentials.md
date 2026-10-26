# Exponentials and Logarithms

The exponential map is the bridge connecting Lie algebras to Lie groups. In geometric algebra, it takes bivectors (infinitesimal rotations) to rotors (finite rotations).

## The Exponential Map

For a k-vector $b$ where $b^2 = λ$ is scalar, the exponential follows from the Taylor series:

```math
e^{b} = 1 + b + \frac{b^2}{2!} + \frac{b^3}{3!} + \cdots
```

This series telescopes into a closed form based on the sign of $b^2$:

```math
e^{b} = \begin{cases}
\cosh\sqrt{λ} + \dfrac{b}{\sqrt{λ}} \sinh\sqrt{λ} & \text{if } λ > 0 \text{ (hyperbolic)} \\
\cos\sqrt{-λ} + \dfrac{b}{\sqrt{-λ}} \sin\sqrt{-λ} & \text{if } λ < 0 \text{ (trigonometric)} \\
1 + b & \text{if } λ = 0 \text{ (nilpotent)}
\end{cases}
```

## Metric Signature Dependence

The sign of $b^2$ depends on the metric signature:

- **Euclidean**: Bivectors square to negative $\Rightarrow$ trigonometric exponentials (rotations)
- **Minkowski**: Some bivectors square positive (boosts), others negative (rotations)
- **Degenerate (PGA)**: Some bivectors square to zero (translations)

```python
from morphis.elements import basis_vectors, euclidean_metric
from morphis.operations import geometric

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

b = e1 ^ e2  # Bivector
b_squared = geometric(b, b)
# In Euclidean: b^2 = -|b|^2 < 0 (trigonometric case)
```

## Computational Efficiency

Unlike matrix exponentials which require Padé approximation or eigendecomposition, k-vector exponentials reduce to **closed-form scalar operations**.

The Taylor series naturally separates:

```math
e^{b} = \underbrace{\left(1 + \frac{λ}{2!} + \frac{λ^2}{4!} + \cdots\right)}_{\cos\sqrt{-λ}} + b \underbrace{\left(1 + \frac{λ}{3!} + \frac{λ^2}{5!} + \cdots\right)}_{\sin(\sqrt{-λ})/\sqrt{-λ}}
```

The computation requires:
1. One geometric product to compute $b^2$
2. Extract scalar $λ$
3. Compute scalar trig/hyperbolic functions
4. Scale the k-vector

```python
from morphis.operations import exp_vector

# Exponential of a vector
R = exp_vector(b)  # Returns MultiVector
```

## Rotors from Bivectors

A **rotor** for rotation by angle $θ$ in the plane defined by unit bivector $\hat{b}$:

```math
R = e^{-\hat{b}θ/2} = \cos(θ/2) - \sin(θ/2) \hat{b}
```

The **half-angle** appears because the sandwich product applies the rotation twice:

```math
v' = R v \tilde{R}
```

```python
from morphis.transforms import rotor
from numpy import pi

# Create rotor for 90-degree rotation in xy-plane
b = (e1 ^ e2).unit()  # Unit bivector
R = rotor(b, pi/2)         # Angle in radians

# R has grades {0, 2}
R[0]  # cos(pi/4) ≈ 0.707
R[2]  # -sin(pi/4) * b

# Apply rotation
v_rotated = R * e1 * ~R  # e1 becomes e2
```

## Non-Simple Bivectors

From four dimensions on, a bivector need not be a blade. The sum of two rotations in orthogonal planes,

```math
b = θ_1 \mathbf{e}_{12} + θ_2 \mathbf{e}_{34}
```

squares to $b^2 = -θ_1^2 - θ_2^2 + 2 θ_1 θ_2 \mathbf{e}_{1234}$, which is not a scalar, so the closed form above does not apply. In four and five dimensions every bivector splits as $b = b_1 + b_2$ into simple bivectors in orthogonal planes, which commute, with $b_k^2 = λ_k$ scalars. Two invariants of $b$ fix the $λ_k$:

```math
\begin{aligned}
\langle b^2 \rangle_0 &= λ_1 + λ_2 \\
w = \tfrac{1}{2} b \wedge b &= b_1 b_2 \\
w^2 &= λ_1 λ_2
\end{aligned}
```

so the $λ_k$ are the roots of $λ^2 - \langle b^2 \rangle_0 λ + w^2 = 0$, and each part is $b_k = (λ_k b - b w) / (λ_k - λ_l)$. Because the parts commute, $e^b = e^{b_1} e^{b_2}$. Writing $c_k = \cos\sqrt{-λ_k}$ and $s_k = \sin\sqrt{-λ_k} / \sqrt{-λ_k}$, which are entire functions of $λ_k$ (cosh and sinh for $λ_k > 0$, and 1 at $λ_k = 0$), the product expressed through $b$ and $w$ alone is:

```math
e^b = c_1 c_2 + s_1 s_2 w + \tfrac{1}{2} (c_2 s_1 + c_1 s_2) b + \tfrac{1}{2} D \left( \langle b^2 \rangle_0 b - 2 b w \right)
```

with the divided difference $D = (c_2 s_1 - c_1 s_2) / (λ_1 - λ_2)$. The split itself is never formed. One formula covers two rotations, a rotation with a boost in spacetime, and a screw in PGA, where one $λ_k$ is zero and its factor is $1 + b_k$. A simple bivector has $w = 0$ and reduces to $\cos + b \sin$. This is the invariant decomposition of Roelfs and De Keninck.

When the roots coincide, as for the isoclinic rotation $b = θ (\mathbf{e}_{12} + \mathbf{e}_{34})$ that turns every vector by the same angle, any split serves and $\langle b^2 \rangle_0 b - 2 b w$ vanishes, so $D$ is taken at its finite limit. There the exponential is:

```math
e^{θ (\mathbf{e}_{12} + \mathbf{e}_{34})} = \cos^2 θ + \sin θ \cos θ (\mathbf{e}_{12} + \mathbf{e}_{34}) + \sin^2 θ \mathbf{e}_{1234}
```

A rotor of a non-simple bivector therefore has a grade-4 part, and the sandwich $R v \tilde{R}$ applies it as before:

```python
from numpy import pi, sqrt
from morphis.elements import basis_vectors, euclidean_metric
from morphis.transforms import rotor

g = euclidean_metric(4)
e1, e2, e3, e4 = basis_vectors(g)

b = ((e1 ^ e4) + (e2 ^ e3)) / sqrt(2)   # isoclinic: equal turns in e_14 and e_23
R = rotor(b, pi / 3)                   # exp(-b π/6), grades {0, 2, 4}
turned = R * e1 * ~R                   # e1 turned by π/(3√2) in the e_14 plane
```

`exp_vector` sends every bivector to `exp_bivector`, which applies this formula in four and five dimensions and the simple form elsewhere. Six or more dimensions allow three or more invariant planes; there a non-simple bivector raises `NotImplementedError`. `log_versor` inverts simple rotors only.

## The Logarithm

The **logarithm** extracts the generator from a versor. For rotor $R = a + b$ where $a$ is the scalar part and $b$ is the bivector part:

```math
\log R = \arctan2(|b|, a) \cdot \frac{b}{|b|}
```

The result is a bivector whose:
- Direction defines the rotation plane
- Magnitude equals half the rotation angle

```python
from morphis.operations import log_versor

# Extract bivector from rotor
b_generator = log_versor(R)

# This bivector can recreate the rotor
R_recovered = exp_vector(b_generator)
```

## Spherical Linear Interpolation (Slerp)

Smooth interpolation between rotors uses the exponential/logarithm:

```math
R(t) = R_0 e^{t \log(R_0^{-1} R_1)}
```

This **slerp** (spherical linear interpolation):
- Maintains constant angular velocity
- Preserves the group structure (result is always a rotor)
- Produces geodesic paths on the rotation manifold

```python
from morphis.operations import slerp

# Interpolate between two rotors
R_half = slerp(R0, R1, t=0.5)

# Animation: interpolate from t=0 to t=1
for t in linspace(0, 1, 100):
    R_current = slerp(R_start, R_end, t)
    frame = R_current * frame_initial * ~R_current
```

Linear interpolation of rotor components, by contrast, doesn't preserve unit magnitude and causes acceleration artifacts.

## Composition via Addition

An elegant property: rotor composition approximately equals bivector addition:

```math
e^{b_1} e^{b_2} \approx e^{b_1 + b_2}
```

**Exact when**: $[b_1, b_2] = 0$ (commuting bivectors)

**General case**: Governed by the Baker-Campbell-Hausdorff formula

This provides an additive parameterization of rotations, useful for optimization and control.

## Translators (PGA)

In Projective Geometric Algebra, degenerate bivectors square to zero:

```math
\mathbf{t}^2 = 0
```

The exponential truncates to a linear term:

```math
T = e^{-\mathbf{t}/2} = 1 - \frac{1}{2}\mathbf{t}
```

This **translator** implements pure translation via the sandwich product.

```python
from morphis.transforms import translator

# Create translator
T = translator(direction)

# Translators compose additively
T_composed = T1 * T2  # = translator(d1 + d2)
```

## Motors

A **motor** combines rotation and translation:

```math
M = RT
```

or equivalently via the exponential of a line:

```math
M = e^{-\ell θ/2}
```

where $\ell$ is a PGA line (bivector + bivector).

```python
from morphis.transforms import rotor, translator, rotation_about_point

# Compose rotation and translation
M = T * R  # Translate after rotate

# Or rotate about a point directly
M = rotation_about_point(center, b, angle)
```

## Connection to Lie Theory

The relationship between bivectors and rotors is a specific instance of the exponential map from a Lie algebra to its Lie group:

```math
\exp: \mathfrak{so}(n) \to \text{SO}(n)
```

The bivector space $\bigwedge^2 V$ with the commutator product is the Lie algebra $\mathfrak{so}(V)$:

```math
[b_1, b_2] = \frac{1}{2}(b_1 b_2 - b_2 b_1)
```

This Lie algebra structure governs:
- Composition of rotations
- Infinitesimal generators
- Angular velocity representation

## Summary

| Object | Exponential Input | Result |
|--------|------------------|--------|
| Euclidean bivector | $e^{-bθ/2}$ | Rotor (rotation) |
| Minkowski timelike bivector | $e^{bφ/2}$ | Boost |
| PGA degenerate bivector | $e^{-t/2}$ | Translator |
| PGA line | $e^{-Lθ/2}$ | Motor (screw motion) |
