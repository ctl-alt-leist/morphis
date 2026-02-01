# Exponentials and Logarithms

The exponential map is the bridge connecting Lie algebras to Lie groups. In geometric algebra, it takes bivectors (infinitesimal rotations) to rotors (finite rotations).

## The Exponential Map

For a k-vector $b$ where $b^2 = \lambda$ is scalar, the exponential follows from the Taylor series:

$$
e^{b} = 1 + b + \frac{b^2}{2!} + \frac{b^3}{3!} + \cdots
$$

This series telescopes into a closed form based on the sign of $b^2$:

$$
e^{b} = \begin{cases}
\cosh\sqrt{\lambda} + \dfrac{b}{\sqrt{\lambda}} \sinh\sqrt{\lambda} & \text{if } \lambda > 0 \text{ (hyperbolic)} \\[1em]
\cos\sqrt{-\lambda} + \dfrac{b}{\sqrt{-\lambda}} \sin\sqrt{-\lambda} & \text{if } \lambda < 0 \text{ (trigonometric)} \\[1em]
1 + b & \text{if } \lambda = 0 \text{ (nilpotent)}
\end{cases}
$$

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

$$
e^{b} = \underbrace{\left(1 + \frac{\lambda}{2!} + \frac{\lambda^2}{4!} + \cdots\right)}_{\cos\sqrt{-\lambda}} + b \underbrace{\left(1 + \frac{\lambda}{3!} + \frac{\lambda^2}{5!} + \cdots\right)}_{\sin(\sqrt{-\lambda})/\sqrt{-\lambda}}
$$

The computation requires:
1. One geometric product to compute $b^2$
2. Extract scalar $\lambda$
3. Compute scalar trig/hyperbolic functions
4. Scale the k-vector

```python
from morphis.operations import exp_vector

# Exponential of a vector
R = exp_vector(b)  # Returns MultiVector
```

## Rotors from Bivectors

A **rotor** for rotation by angle $\theta$ in the plane defined by unit bivector $\hat{b}$:

$$
R = e^{-\hat{b}\theta/2} = \cos(\theta/2) - \sin(\theta/2) \, \hat{b}
$$

The **half-angle** appears because the sandwich product applies the rotation twice:

$$
v' = R v \tilde{R}
$$

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

## The Logarithm

The **logarithm** extracts the generator from a versor. For rotor $R = a + b$ where $a$ is the scalar part and $b$ is the bivector part:

$$
\log R = \arctan2(|b|, a) \cdot \frac{b}{|b|}
$$

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

$$R(t) = R_0 \, e^{t \log(R_0^{-1} R_1)}$$

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

$$
e^{b_1} e^{b_2} \approx e^{b_1 + b_2}
$$

**Exact when**: $[b_1, b_2] = 0$ (commuting bivectors)

**General case**: Governed by the Baker-Campbell-Hausdorff formula

This provides an additive parameterization of rotations, useful for optimization and control.

## Translators (PGA)

In Projective Geometric Algebra, degenerate bivectors square to zero:

$$\mathbf{t}^2 = 0$$

The exponential truncates to a linear term:

$$T = e^{-\mathbf{t}/2} = 1 - \frac{1}{2}\mathbf{t}$$

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

$$M = RT$$

or equivalently via the exponential of a line:

$$M = e^{-L\theta/2}$$

where $L$ is a PGA line (bivector + bivector).

```python
from morphis.transforms import rotor, translator, rotation_about_point

# Compose rotation and translation
M = T * R  # Translate after rotate

# Or rotate about a point directly
M = rotation_about_point(center, b, angle)
```

## Connection to Lie Theory

The relationship between bivectors and rotors is a specific instance of the exponential map from a Lie algebra to its Lie group:

$$
\exp: \mathfrak{so}(n) \to \text{SO}(n)
$$

The bivector space $\bigwedge^2 V$ with the commutator product is the Lie algebra $\mathfrak{so}(V)$:

$$
[b_1, b_2] = \frac{1}{2}(b_1 b_2 - b_2 b_1)
$$

This Lie algebra structure governs:
- Composition of rotations
- Infinitesimal generators
- Angular velocity representation

## Summary

| Object | Exponential Input | Result |
|--------|------------------|--------|
| Euclidean bivector | $e^{-b\theta/2}$ | Rotor (rotation) |
| Minkowski timelike bivector | $e^{b\phi/2}$ | Boost |
| PGA degenerate bivector | $e^{-t/2}$ | Translator |
| PGA line | $e^{-L\theta/2}$ | Motor (screw motion) |
