# Transforms: Mathematical Properties

Mathematical foundations for rotors, translators, motors, and projective geometry operations.

## Rotors

A rotor represents rotation via the exponential map from bivector to transformation.

### Exponential Form

For unit bivector $B$ with $B^2 = -1$ and angle $θ$:

$$
R = e^{-Bθ/2} = \cos(θ/2) - \sin(θ/2) B
$$

### Multivector Structure

$$
R = R_0 \mathbf{1} + R^{mn} \mathbf{e}_{mn}
$$

where:
- $R_0 = \cos(θ/2)$
- $R^{mn} = -\sin(θ/2) B^{mn}$

### Versor Property

$$
R \tilde{R} = 1
$$

Proof:

$$
R \tilde{R} = \cos^2(θ/2) + \sin^2(θ/2) = 1
$$

### Inverse

$$
R^{-1} = \tilde{R} = \cos(θ/2) + \sin(θ/2) B
$$

### Identity

$$
R(0) = 1
$$

### Non-Simple Bivectors

In four and five dimensions $B$ need not be simple, and `exp_bivector` evaluates $e^B$ in closed form through $\langle B^2 \rangle_0$ and $W = \tfrac{1}{2} B \wedge B$. Each case is checked against the defining series $\sum_n B^n / n!$ in the geometric product:

- a random bivector of $\mathbb{R}^4$ and of $\mathbb{R}^5$
- the isoclinic $θ (\mathbf{e}_{12} + \mathbf{e}_{34})$, where also $\langle e^B \rangle_0 = \cos^2 θ$ and $\langle e^B \rangle_4 = \sin^2 θ \mathbf{e}_{1234}$
- nearly isoclinic bivectors, whose roots differ by $10^{-5}$ to $10^{-12}$
- a bivector of size $10^{-5}$, to $10^{-15}$
- random bivectors of spacetime in both conventions
- a PGA screw $0.8 \mathbf{e}_{12} + 0.5 \mathbf{e}_{03} + 0.3 \mathbf{e}_{01}$
- a lot of shape $(3, 2)$

A simple bivector gives $\cos + B \sin$ with no grade-4 part, also in six dimensions, where a non-simple bivector raises `NotImplementedError`. For $B = 0.9 \mathbf{u}_1 \wedge \mathbf{u}_2 - 0.4 \mathbf{u}_3 \wedge \mathbf{u}_4$ in a random orthonormal frame, `rotor(B, 1)` turns vectors as the product of the two plane rotors, and $R \tilde{R} = 1$.

## Sandwich Product

Rotors act via:

$$
\mathbf{p}' = R \mathbf{p} \tilde{R}
$$

### Rodrigues Formula

For rotation in plane $B$ by angle $θ$:

$$
x'^m = x^m + \sin(θ) B^{mp} x_p + (1 - \cos(θ)) B^{mp} B_p^{\ q} x_q
$$

### Rotation in xy-Plane

With $B = \mathbf{e}_{12}$:

$$
\begin{align}
x' &= x \cos θ - y \sin θ \\
y' &= x \sin θ + y \cos θ \\
z' &= z
\end{align}
$$

### Specific Rotations

90 degrees ($θ = π/2$):

$$
(1, 0, 0) \mapsto (0, 1, 0)
$$

180 degrees ($θ = π$):

$$
(x, y, z) \mapsto (-x, -y, z)
$$

### Plane Invariance

$$
R B \tilde{R} = B
$$

## Translators

A translator represents pure translation in PGA.

### Exponential Form

$$
T = e^{-\mathbf{t}/2} = 1 - \frac{1}{2} t^m \mathbf{e}_{0m}
$$

where $\mathbf{t} = t^m \mathbf{e}_{0m}$ is the degenerate bivector.

### Truncation

Degenerate bivectors square to zero:

$$
\mathbf{t}^2 = 0
$$

so the exponential truncates to linear term.

### Translation Action

$$
\mathbf{p}' = T \mathbf{p} \tilde{T}
$$

For point $\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$:

$$
\mathbf{p}' = \mathbf{e}_0 + (x^m + t^m) \mathbf{e}_m
$$

### Composition

Translators compose additively:

$$
T_{\mathbf{t}} T_{\mathbf{s}} = T_{\mathbf{t} + \mathbf{s}}
$$

### Commutativity

$$
T_{\mathbf{t}} T_{\mathbf{s}} = T_{\mathbf{s}} T_{\mathbf{t}}
$$

### Versor Property

$$
T \tilde{T} = 1
$$

## Motors

Motors combine rotation and translation.

### Composition

$$
M = RT
$$

or via line exponential:

$$
M = e^{-Lθ/2}
$$

### Rotation About Arbitrary Center

$$
M = T_{\mathbf{c}} R T_{-\mathbf{c}}
$$

### Center Invariance

$$
M \mathbf{p}_c \tilde{M} = \mathbf{p}_c
$$

### Non-Commutativity

$$
RT \neq TR
$$

"Rotate then translate" differs from "translate then rotate."

### Associativity

$$
(M_3 M_2) M_1 = M_3 (M_2 M_1)
$$

### Closure

- Rotor $\times$ Rotor = Rotor
- Translator $\times$ Translator = Translator
- Rotor $\times$ Translator = Motor

All maintain versor property: $M \tilde{M} = 1$.

## Geometric Invariants

### Distance Preservation

$$
d(\mathbf{p}'_1, \mathbf{p}'_2) = d(\mathbf{p}_1, \mathbf{p}_2)
$$

### Angle Preservation

$$
\cos θ' = \cos θ
$$

### Orientation Preservation

$$
\det(M) = +1
$$

## Reflections

### Formula

$$
\mathbf{v}' = -\mathbf{n}\mathbf{v}\mathbf{n}
$$

### Properties

Involution (twice = identity):

$$
-\mathbf{n}(-\mathbf{n}\mathbf{v}\mathbf{n})\mathbf{n} = \mathbf{v}
$$

Magnitude preservation:

$$
|\mathbf{v}'| = |\mathbf{v}|
$$

## Versors

### From Reflections

$$
V = \mathbf{n}_k \cdots \mathbf{n}_2 \mathbf{n}_1
$$

### Two Reflections = Rotation

$$
\mathbf{n}_2 \mathbf{n}_1 = R_{2θ}
$$

where $θ$ is angle between reflection planes.

### Cartan-Dieudonne

Any orthogonal transformation in $d$ dimensions factors as at most $d$ reflections.

## Projective Geometry (PGA)

### Point Embedding

$$
\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m
$$

### Direction Embedding

$$
\mathbf{d} = v^m \mathbf{e}_m
$$

(zero ideal component)

### Weight and Bulk

$$
w(\mathbf{p}) = p^0
$$

$$
\text{bulk}(\mathbf{p}) = (p^1, \ldots, p^d)
$$

### Euclidean Projection

$$
\text{euclidean}(\mathbf{p}) = \frac{\text{bulk}(\mathbf{p})}{w(\mathbf{p})}
$$

### Geometric Constructors

Line through points:

$$
\ell = \mathbf{p} \wedge \mathbf{q}
$$

Plane through points:

$$
π = \mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r}
$$

### Distance Functions

Point-to-point:

$$
d(\mathbf{p}, \mathbf{q}) = \sqrt{g_{mn} (x_q^m - x_p^m)(x_q^n - x_p^n)}
$$

Point-to-line:

$$
d(\mathbf{p}, \ell) = \frac{|\mathbf{p} \wedge \ell|}{|\ell|}
$$

### Incidence Predicates

Collinearity:

$$
\mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r} = 0
$$

Coplanarity:

$$
\mathbf{p} \wedge \mathbf{q} \wedge \mathbf{r} \wedge \mathbf{s} = 0
$$

Point on line:

$$
\mathbf{p} \wedge \ell = 0
$$

## Higher Dimensions

### Rotation Planes

In $d$ dimensions: $\binom{d}{2}$ independent rotation planes.

4D bivector basis:

$$
\mathbf{e}_{12}, \mathbf{e}_{13}, \mathbf{e}_{14}, \mathbf{e}_{23}, \mathbf{e}_{24}, \mathbf{e}_{34}
$$

### Component Counts

Motor components: $1 + \binom{d + 1}{2}$

- 3D: $1 + 6 = 7$
- 4D: $1 + 10 = 11$

### Half-Angle Convention

Exponential uses half-angles so sandwich product gives full rotation:

$$
R \mathbf{v} \tilde{R} \text{ rotates by } θ
$$
