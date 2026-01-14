# Motors: Mathematical Properties for Testing

## Rotors and the Exponential Map

A rotor in projective geometric algebra represents a pure rotation about the origin. The exponential map from the Lie algebra to the Lie group takes a bivector $B$ and angle $\theta$ to produce:

$$M = e^{-B\theta/2} = \cos(\theta/2) - \sin(\theta/2) B$$

where $B$ is a unit Euclidean bivector satisfying $B^2 = -1$. The bivector $B$ specifies the plane of rotation, while $\theta$ gives the rotation angle within that plane.

The multivector structure breaks down into exactly two grades:

$$M = M_0 \mathbf{1} + M^{mn} \mathbf{e}_{mn}$$

The scalar part captures the "amount of rotation":

$$M_0 = \cos(\theta/2)$$

while the bivector part encodes both the rotation plane and the complementary sine component:

$$M^{mn} = -\sin(\theta/2) B^{mn}$$

This half-angle formulation ensures that applying the rotor via the sandwich product produces the full rotation angle $\theta$. The factor of $-1$ in the bivector term comes from the exponential series and the convention that $e^{-B\theta/2}$ produces counterclockwise rotation in the oriented plane $B$.

For a rotation of $\pi/2$ radians (90 degrees) in the $xy$-plane with $B = \mathbf{e}_{12}$:

$$M = \cos(\pi/4) - \sin(\pi/4) \mathbf{e}_{12} = \frac{\sqrt{2}}{2} - \frac{\sqrt{2}}{2} \mathbf{e}_{12}$$

For a full rotation of $\pi$ radians (180 degrees):

$$M = \cos(\pi/2) - \sin(\pi/2) B = 0 - B = -B$$

The rotor is purely bivector, reflecting the fact that 180-degree rotations square to $-1$ in the geometric algebra.

## The Versor Property

Rotors belong to the special class of multivectors called versorsâ€"those satisfying the normalization condition:

$$M \tilde{M} = 1$$

where $\tilde{M}$ denotes the reverse operation. For the rotor $M = \cos(\theta/2) - \sin(\theta/2) B$, the reverse is:

$$\tilde{M} = \cos(\theta/2) + \sin(\theta/2) B$$

The geometric product of these two yields:

$$
\begin{align}
M \tilde{M} &= \left(\cos(\theta/2) - \sin(\theta/2) B\right)\left(\cos(\theta/2) + \sin(\theta/2) B\right) \\ \\
&= \cos^2(\theta/2) + \cos(\theta/2)\sin(\theta/2) B - \sin(\theta/2)\cos(\theta/2) B - \sin^2(\theta/2) B^2 \\ \\
&= \cos^2(\theta/2) - \sin^2(\theta/2) B^2 \\ \\
&= \cos^2(\theta/2) + \sin^2(\theta/2) \\ \\
&= 1
\end{align}
$$

The cross terms vanish due to anticommutativity of the bivector with itself, while $B^2 = -1$ converts the negative sine-squared term to positive. The result is a pure scalar equal to unity, confirming the versor property.

This normalization ensures that rotors preserve magnitudes under the sandwich product transformationâ€"a geometric object's "size" remains unchanged when rotated.

## Inverse Rotors and Identity

For any rotor, the inverse equals the reverse:

$$M^{-1} = \tilde{M}$$

This follows immediately from the versor property $M \tilde{M} = 1$. Geometrically, if $M$ rotates by angle $\theta$, then $\tilde{M}$ rotates by angle $-\theta$:

$$\tilde{M} = \cos(\theta/2) + \sin(\theta/2) B = e^{B\theta/2}$$

Composing a rotor with its inverse returns the identity:

$$M M^{-1} = M \tilde{M} = 1$$

The identity rotor corresponds to zero rotation angle:

$$M(0) = \cos(0) - \sin(0) B = 1 - 0 = 1$$

This is simply the scalar unity, with no bivector component. Applying the identity rotor to any geometric object leaves it unchanged.

## The Sandwich Product Transformation

Rotors act on PGA points through the sandwich product:

$$\mathbf{p}' = M \mathbf{p} \tilde{M}$$

A point in projective geometric algebra has the form $\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$, where $\mathbf{e}_0$ represents the ideal point at infinity and the Euclidean components $x^m \mathbf{e}_m$ encode the affine position.

The sandwich product transforms the Euclidean components while leaving the ideal component unchanged, producing:

$$\mathbf{p}' = \mathbf{e}_0 + (x')^m \mathbf{e}_m$$

The coordinate transformation follows the Rodrigues rotation formula, which can be expressed through bivector components:

$$x'^m = x^m + \sin(\theta) B^{mp} x_p + (1 - \cos(\theta)) B^{mp} B_p^{\ q} x_q$$

For rotation in the $xy$-plane using $B = \mathbf{e}_{12}$ (with $B^{12} = 1$, all other components zero), the transformation becomes:

$$
\begin{align}
x' &= x \cos\theta - y \sin\theta \\ \\
y' &= x \sin\theta + y \cos\theta \\ \\
z' &= z
\end{align}
$$

This is the familiar 2D rotation matrix acting on the $xy$ coordinates while leaving $z$ untouched.

Consider specific examples. A 90-degree rotation ($\theta = \pi/2$) maps:

$$(1, 0, 0) \mapsto (0, 1, 0)$$

$$(0, 1, 0) \mapsto (-1, 0, 0)$$

$$(0, 0, 1) \mapsto (0, 0, 1)$$

The first two points rotate within the $xy$-plane, while the third point lies perpendicular to that plane and remains fixed.

A 180-degree rotation ($\theta = \pi$) in the $xy$-plane maps:

$$(x, y, z) \mapsto (-x, -y, z)$$

Points reflect through the origin within the rotation plane.

The rotation plane itself is invariant under the rotor action in a specific sense: the bivector $B$ transforms as:

$$M B \tilde{M} = B$$

Any vector lying within the plane spanned by $B$ rotates, but the plane itself as an oriented 2D subspace remains unchanged.

## Translators and Degenerate Bivectors

A translator represents pure translation in projective geometric algebra. Unlike Euclidean geometric algebra where translations require separate machinery, PGA encodes translations naturally through degenerate bivectorsâ€"those involving the ideal point $\mathbf{e}_0$.

The exponential map for translation uses a degenerate bivector:

$$M = e^{-\mathbf{t}/2} = 1 - \frac{1}{2} t^m \mathbf{e}_{0m}$$

where $\mathbf{t} = t^m \mathbf{e}_{0m}$ encodes the displacement vector. Expanding the exponential:

$$e^{-\mathbf{t}/2} = 1 - \frac{\mathbf{t}}{2} + \frac{\mathbf{t}^2}{8} - \frac{\mathbf{t}^3}{48} + \cdots$$

The series truncates after the linear term because degenerate bivectors square to zero:

$$\mathbf{t}^2 = (t^m \mathbf{e}_{0m})(t^n \mathbf{e}_{0n}) = t^m t^n \mathbf{e}_{0m} \mathbf{e}_{0n}$$

Since $\mathbf{e}_0$ anticommutes with itself ($\mathbf{e}_0 \mathbf{e}_0 = 0$ in the degenerate metric), all terms vanish identically. The exponential simplifies to:

$$M = 1 - \frac{1}{2} t^m \mathbf{e}_{0m}$$

This translator contains only two grades: a scalar part equal to 1, and a purely degenerate bivector part. The Euclidean bivector components $\mathbf{e}_{mn}$ (with $m, n \geq 1$) are all zero.

## Translator Action on Points

Applying a translator via the sandwich product shifts points by the displacement vector:

$$\mathbf{p}' = M \mathbf{p} \tilde{M}$$

For point $\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$ translated by displacement $t^m$:

$$\mathbf{p}' = \mathbf{e}_0 + (x^m + t^m) \mathbf{e}_m$$

The coordinate transformation is pure additive translation:

$$x'^m = x^m + t^m$$

This follows from the geometric product structure. The computation involves:

$$
\begin{align}
M \mathbf{p} \tilde{M} &= \left(1 - \frac{1}{2} t^n \mathbf{e}_{0n}\right) (\mathbf{e}_0 + x^m \mathbf{e}_m) \left(1 + \frac{1}{2} t^p \mathbf{e}_{0p}\right) \\ \\
&= \mathbf{e}_0 + x^m \mathbf{e}_m + t^m \mathbf{e}_m
\end{align}
$$

after applying the PGA product rules and using $\mathbf{e}_{0m} \mathbf{e}_m = \mathbf{1}$ from the metric structure.

Consider translating the origin by $(1, 2, 3)$. The origin point is $\mathbf{p}_0 = \mathbf{e}_0$, and after translation:

$$\mathbf{p}_0' = \mathbf{e}_0 + 1 \cdot \mathbf{e}_1 + 2 \cdot \mathbf{e}_2 + 3 \cdot \mathbf{e}_3$$

For an arbitrary point at $(5, -3, 2)$ translated by $(-2, 1, 0)$:

$$
\begin{align}
\mathbf{p} &= \mathbf{e}_0 + 5\mathbf{e}_1 - 3\mathbf{e}_2 + 2\mathbf{e}_3 \\ \\
\mathbf{p}' &= \mathbf{e}_0 + 3\mathbf{e}_1 - 2\mathbf{e}_2 + 2\mathbf{e}_3
\end{align}
$$

The result is $(3, -2, 2)$, confirming the coordinate-wise addition.

## Translator Composition and Commutativity

Unlike rotations, translators compose additively and commute. For two translators with displacements $\mathbf{s}$ and $\mathbf{t}$:

$$M_{\mathbf{t}} M_{\mathbf{s}} = M_{\mathbf{t} + \mathbf{s}}$$

Expanding the left side:

$$
\begin{align}
M_{\mathbf{t}} M_{\mathbf{s}} &= \left(1 - \frac{\mathbf{t}}{2}\right)\left(1 - \frac{\mathbf{s}}{2}\right) \\ \\
&= 1 - \frac{\mathbf{t}}{2} - \frac{\mathbf{s}}{2} + \frac{\mathbf{t} \mathbf{s}}{4}
\end{align}
$$

The product of two degenerate bivectors vanishes:

$$\mathbf{t} \mathbf{s} = (t^m \mathbf{e}_{0m})(s^n \mathbf{e}_{0n}) = t^m s^n \mathbf{e}_{0m} \mathbf{e}_{0n} = 0$$

This uses the fact that $\mathbf{e}_0$ squares to zero and the antisymmetry $\mathbf{e}_{0m} \mathbf{e}_{0n} = -\mathbf{e}_{0n} \mathbf{e}_{0m}$ makes the symmetric sum vanish. Therefore:

$$M_{\mathbf{t}} M_{\mathbf{s}} = 1 - \frac{\mathbf{t} + \mathbf{s}}{2} = M_{\mathbf{t} + \mathbf{s}}$$

Translators form an abelian subgroup under composition:

$$M_{\mathbf{t}} M_{\mathbf{s}} = M_{\mathbf{s}} M_{\mathbf{t}}$$

Translating by $(1, 0, 0)$ then by $(0, 2, 0)$ produces the same result as translating by $(0, 2, 0)$ then by $(1, 0, 0)$â€"both yield a net displacement of $(1, 2, 0)$.

## Translator Versor Property

Like rotors, translators satisfy the versor normalization:

$$M \tilde{M} = 1$$

The reverse of a translator is:

$$\tilde{M} = 1 + \frac{1}{2} t^m \mathbf{e}_{0m}$$

Computing the product:

$$
\begin{align}
M \tilde{M} &= \left(1 - \frac{\mathbf{t}}{2}\right)\left(1 + \frac{\mathbf{t}}{2}\right) \\ \\
&= 1 - \frac{\mathbf{t}^2}{4} \\ \\
&= 1
\end{align}
$$

since $\mathbf{t}^2 = 0$ for degenerate bivectors. This versor property ensures translators preserve the metric structure during transformations.

## Rotation About Arbitrary Centers

A rotation about an arbitrary center point $\mathbf{c}$ combines translation and rotation through the decomposition:

$$M = T_{\mathbf{c}} R T_{-\mathbf{c}}$$

Here $T_{\mathbf{c}}$ translates by displacement $\mathbf{c}$, $R$ rotates about the origin, and $T_{-\mathbf{c}}$ translates back by $-\mathbf{c}$. This three-step process:

1. Moves the center to the origin
2. Rotates about the origin
3. Moves back to the original center

The explicit motor form is:

$$M = \left(1 - \frac{c^m \mathbf{e}_{0m}}{2}\right) \left(\cos(\theta/2) - \sin(\theta/2) B\right) \left(1 + \frac{c^m \mathbf{e}_{0m}}{2}\right)$$

Expanding this product yields a general motor with both Euclidean and degenerate bivector components.

The center point itself remains fixed under this transformation:

$$M \mathbf{p}_c \tilde{M} = \mathbf{p}_c$$

where $\mathbf{p}_c = \mathbf{e}_0 + c^m \mathbf{e}_m$ is the center encoded as a PGA point. By construction, the translators map $\mathbf{p}_c \mapsto \mathbf{e}_0$, which is invariant under rotation about the origin, then back to $\mathbf{p}_c$.

Consider rotating by 90 degrees about the point $(1, 0, 0)$ in the $xy$-plane. A test point at $(2, 0, 0)$ lies at relative position $(1, 0, 0)$ from the center. After rotation, this becomes $(0, 1, 0)$ relative to the center, or $(1, 1, 0)$ in global coordinates:

$$
\begin{align}
(2, 0, 0) - (1, 0, 0) &= (1, 0, 0) \\ \\
\text{rotate 90°:} \quad (1, 0, 0) &\mapsto (0, 1, 0) \\ \\
(0, 1, 0) + (1, 0, 0) &= (1, 1, 0)
\end{align}
$$

The decomposition formula $M = T_{\mathbf{c}} R T_{-\mathbf{c}}$ is equivalent to the line exponential:

$$M = e^{-L\theta/2}$$

where $L = \mathbf{p}_c \wedge B$ is the Plücker line encoding both the rotation plane $B$ and the center point $\mathbf{p}_c$. This line is a grade-3 object in the 4D homogeneous space of 3D PGA:

$$L = \mathbf{p}_c \wedge B = (\mathbf{e}_0 + c^m \mathbf{e}_m) \wedge (B^{nq} \mathbf{e}_{nq})$$

Expanding the wedge product produces terms like $\mathbf{e}_{0nq}$ (degenerate trivectors) and $c^m B^{nq} \mathbf{e}_{mnq}$ (Euclidean trivectors). Together these encode the full geometric structure of the rotation: the plane, the center, and the axis perpendicular to both (in 3D).

## Motor Composition and Non-Commutativity

Motors compose through the geometric product, producing a new motor. For motors $M_1$ and $M_2$:

$$M_3 = M_2 M_1$$

This composite motor applies transformations right-to-left:

$$M_3(\mathbf{p}) = M_2(M_1(\mathbf{p})) = M_2 M_1 \mathbf{p} \tilde{M}_1 \tilde{M}_2$$

Motor composition is generally non-commutative:

$$M_2 M_1 \neq M_1 M_2$$

The order matters critically. "Rotate then translate" produces a different result than "translate then rotate."

Consider rotating by 90 degrees in the $xy$-plane, then translating by $(1, 0, 0)$. Starting from point $(1, 0, 0)$:

$$
\begin{align}
\text{After rotation:} &\quad (1, 0, 0) \mapsto (0, 1, 0) \\ \\
\text{After translation:} &\quad (0, 1, 0) \mapsto (1, 1, 0)
\end{align}
$$

Now reverse the order: translate by $(1, 0, 0)$, then rotate by 90 degrees:

$$
\begin{align}
\text{After translation:} &\quad (1, 0, 0) \mapsto (2, 0, 0) \\ \\
\text{After rotation:} &\quad (2, 0, 0) \mapsto (0, 2, 0)
\end{align}
$$

The final positions $(1, 1, 0)$ and $(0, 2, 0)$ differ, demonstrating non-commutativity.

The commutator measures this difference:

$$[M_2, M_1] = M_2 M_1 M_2^{-1} M_1^{-1}$$

For rotors and translators, this commutator is generally non-trivial, unlike the case of two rotors (which commute if their bivectors commute) or two translators (which always commute).

## Associativity and Group Structure

Despite non-commutativity, motor composition is associative:

$$(M_3 M_2) M_1 = M_3 (M_2 M_1)$$

This follows from the associativity of the geometric product. Parentheses can be grouped arbitrarily without affecting the result, though the left-to-right order must be preserved.

The set of motors forms a group under composition, called the special Euclidean group $SE(d)$ in $d$ dimensions:

**Identity element:** $M_e = 1$ (the scalar unity)

**Inverse:** For any motor $M$, there exists $M^{-1}$ such that $M M^{-1} = M^{-1} M = 1$

**Closure:** The product of two motors is a motor

**Associativity:** Composition is associative

Since motors are versors satisfying $M \tilde{M} = 1$, the inverse equals the reverse:

$$M^{-1} = \tilde{M}$$

This makes inverse computation trivial: simply reverse the order of the geometric product factors and flip the sign of all bivector components.

The group $SE(3)$ has dimension 6: three rotation parameters (one angle per plane) and three translation parameters. In 4D, $SE(4)$ has dimension 10: six rotation planes and four translation directions.

## Versor Closure Under Composition

Composing two versors produces another versor. For motors $M_1$ and $M_2$ satisfying:

$$M_1 \tilde{M}_1 = 1, \quad M_2 \tilde{M}_2 = 1$$

The composite $M_3 = M_2 M_1$ also satisfies:

$$M_3 \tilde{M}_3 = (M_2 M_1)(\tilde{M}_1 \tilde{M}_2) = M_2 (M_1 \tilde{M}_1) \tilde{M}_2 = M_2 \tilde{M}_2 = 1$$

This versor property ensures that all rigid transformations preserve the metric structure.

Specific cases:
- **Rotor × Rotor = Rotor**: Composing rotations about the origin
- **Translator × Translator = Translator**: Composing translations (displacements add)
- **Rotor × Translator = Motor**: General rigid transformation with both rotation and translation

All results maintain the grades {0, 2} structure and satisfy $M \tilde{M} = 1$.

## Distance Preservation (Isometry)

Motors preserve Euclidean distances between points. For any two points $\mathbf{p}_1$ and $\mathbf{p}_2$, the distance after transformation equals the distance before:

$$d(\mathbf{p}'_1, \mathbf{p}'_2) = d(\mathbf{p}_1, \mathbf{p}_2)$$

where $\mathbf{p}'_i = M \mathbf{p}_i \tilde{M}$ and distance is measured in Euclidean coordinates:

$$d(\mathbf{p}_1, \mathbf{p}_2) = \sqrt{g_{mn}(x_1^m - x_2^m)(x_1^n - x_2^n)}$$

This isometry property follows from the versor normalization and the sandwich product structure. The PGA inner product structure is preserved:

$$\mathbf{p}'_1 \cdot \mathbf{p}'_2 = (M \mathbf{p}_1 \tilde{M}) \cdot (M \mathbf{p}_2 \tilde{M}) = \mathbf{p}_1 \cdot \mathbf{p}_2$$

Motors are precisely the transformations that preserve distances while maintaining orientation.

## Angle Preservation and Conformality

Motors preserve angles between vectors. For vectors $\mathbf{u}$ and $\mathbf{v}$ with angle $\theta$:

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}||\mathbf{v}|}$$

After transformation by motor $M$:

$$\cos\theta' = \frac{\mathbf{u}' \cdot \mathbf{v}'}{|\mathbf{u}'||\mathbf{v}'|} = \cos\theta$$

where $\mathbf{u}' = M \mathbf{u} \tilde{M}$ and similarly for $\mathbf{v}$. The angle $\theta' = \theta$ remains unchanged.

This implies orthogonal vectors remain orthogonal. If $\mathbf{u} \cdot \mathbf{v} = 0$, then $\mathbf{u}' \cdot \mathbf{v}' = 0$ after applying any motor. The 90-degree angle between perpendicular directions is preserved.

Motors are thus conformal transformations (angle-preserving) in addition to being isometric (distance-preserving). In fact, for Euclidean space, the rigid motions are exactly those conformal transformations that also preserve a point at infinity.

## Orientation Preservation

Motors preserve the orientation of space. The determinant of the induced linear transformation on Euclidean coordinates is:

$$\det(M) = +1$$

This distinguishes motors (proper rotations) from reflections (improper transformations with $\det = -1$). Motors form the special orthogonal group $SO(d)$ when restricted to rotations about the origin, or the special Euclidean group $SE(d)$ when translations are included.

Handed-ness is preserved: right-handed coordinate systems remain right-handed after applying a motor. The orientation of any $k$-dimensional subspace is maintained under motor action.

## Half-Angle Convention and the Sandwich Product

The exponential formula uses half-angles:

$$M = e^{-B\theta/2}$$

rather than the full angle $\theta$. This convention ensures that the sandwich product produces a rotation by the full angle:

$$M \mathbf{v} \tilde{M} \text{ rotates } \mathbf{v} \text{ by angle } \theta$$

Why does this work? The sandwich product applies the transformation twice:

$$M \mathbf{v} \tilde{M} = e^{-B\theta/2} \mathbf{v} e^{B\theta/2}$$

Through the Baker-Campbell-Hausdorff formula for the adjoint action, this compounds to:

$$\text{Ad}_M(\mathbf{v}) = e^{-B\theta} \mathbf{v} e^{B\theta} = e^{-B\theta}(\mathbf{v})$$

where the final exponential acts on $\mathbf{v}$ via the adjoint representation, producing a rotation by the full angle $\theta$.

The half-angle formulation is a general feature of spin representations. The same phenomenon appears in quantum mechanics, where spinors rotate by half the angle of the physical space they represent.

## Higher-Dimensional Rotations

In $d$-dimensional Euclidean space, there are $\binom{d}{2}$ independent rotation planes corresponding to the $\binom{d}{2}$ bivector basis elements.

In 4D space (using 5D PGA coordinates), the six bivector planes are:

$$\mathbf{e}_{12}, \mathbf{e}_{13}, \mathbf{e}_{14}, \mathbf{e}_{23}, \mathbf{e}_{24}, \mathbf{e}_{34}$$

Each generates an independent rotation. For instance, $\mathbf{e}_{14}$ represents rotation in the $xw$-plane:

$$M = \cos(\theta/2) - \sin(\theta/2) \mathbf{e}_{14}$$

A point $(1, 0, 0, 0)$ rotated by 90 degrees in this plane maps to $(0, 0, 0, 1)$:

$$
\begin{align}
x' &= x \cos\theta - w \sin\theta = 1 \cdot 0 - 0 \cdot 1 = 0 \\ \\
y' &= y = 0 \\ \\
z' &= z = 0 \\ \\
w' &= x \sin\theta + w \cos\theta = 1 \cdot 1 + 0 \cdot 0 = 1
\end{align}
$$

In higher dimensions, rotations always occur in 2-planes, never around axes. The axis-angle representation familiar from 3D is a special case enabled by the Hodge duality between bivectors and vectors in precisely three dimensions.

## Component Counts and Dimensionality

A motor in $d$-dimensional Euclidean space uses $(d+1)$-dimensional PGA and has components:

- Grade-0: 1 component (scalar part)
- Grade-2: $\binom{d+1}{2}$ components (bivector part)

**Total:** $1 + \binom{d+1}{2}$ components

For 3D Euclidean space:

$$1 + \binom{4}{2} = 1 + 6 = 7 \text{ components}$$

For 4D Euclidean space:

$$1 + \binom{5}{2} = 1 + 10 = 11 \text{ components}$$

The dimension of the special Euclidean group $SE(d)$ is:

$$\dim(SE(d)) = \binom{d}{2} + d = \frac{d(d-1)}{2} + d = \frac{d(d+1)}{2}$$

This counts $\binom{d}{2}$ rotation parameters and $d$ translation parameters. For $SE(3)$: $3 + 3 = 6$ dimensions. For $SE(4)$: $6 + 4 = 10$ dimensions.

## Parameterized Families and Animation

A single rotation plane $B$ with varying angle parameter $\theta(t)$ produces a one-parameter family of rotors:

$$M(t) = \cos(\theta(t)/2) - \sin(\theta(t)/2) B$$

For $t \in [0, 1]$ and $\theta(t) = t\theta_{\text{max}}$, this interpolates smoothly from identity ($t=0$) to full rotation ($t=1$):

$$M(0) = 1, \quad M(1) = \cos(\theta_{\text{max}}/2) - \sin(\theta_{\text{max}}/2) B$$

The angular velocity is constant:

$$\frac{dM}{dt} = \frac{d\theta}{dt} \left[-\frac{\sin(\theta/2)}{2} - \frac{\cos(\theta/2)}{2} B\right] = \frac{d\theta}{dt} \left[-\frac{B}{2}\right] M$$

This exponential derivative structure $\frac{dM}{dt} \propto BM$ reflects the Lie algebra generator $B$ acting on the group element $M$.

For a collection of $n$ angles $\theta_1, \ldots, \theta_n$, the motor becomes an array with collection dimension 1 and shape $(n,)$. Each element represents a motor at a specific angle, allowing simultaneous transformation of points along the animation path.

## Euclidean Decomposition

Any motor can be uniquely decomposed into a translation followed by a rotation about the origin:

$$M = T \cdot R$$

where $T$ is a pure translator and $R$ is a pure rotor. This decomposition is not unique in the other direction; $R \cdot T \neq T \cdot R$ in general.

The decomposition extracts:

**Rotor part:** The Euclidean bivector components $M^{mn}$ with $m, n \geq 1$ define the rotation

**Translator part:** The degenerate bivector components $M^{0m}$ define the translation

A general motor thus encodes six degrees of freedom in 3D: three for rotation (the three independent components of a unit bivector times one angle) and three for translation (three displacement components).

## Rigid Body Interpretation

Motors represent exactly the transformations that a rigid body can undergo: rotations and translations, possibly about arbitrary centers. The constraint that motors preserve:
- Distances between all points (rigidity)
- Angles between all directions (no shearing)
- Orientation (no reflection)

makes them the mathematical representation of rigid body motions.

A general rigid body transformation can be described by:
1. A screw axis (line in space)
2. A rotation angle about that axis
3. A translation distance along that axis

This is Chasles' theorem: every rigid body motion is a screw motion. The motor formulation naturally encodes this through the line exponential $M = e^{-L\theta/2}$, where the line $L$ carries both the axis direction and its moment about the origin.

The power of the motor formalism lies in treating all these casesâ€"pure rotations, pure translations, rotations about arbitrary centers, and general screw motionsâ€"within a single algebraic framework using the geometric product and sandwich product transformations.

### 1.3 Inverse Rotor

The inverse of a rotor equals its reverse:

$$M^{-1} = \tilde{M}$$

This follows from the versor normalization $M \tilde{M} = 1$.

**Inverse rotation:** If $M$ rotates by angle $\theta$, then $M^{-1}$ rotates by angle $-\theta$:

$$M^{-1} = \cos(\theta/2) + \sin(\theta/2) B = e^{B\theta/2}$$

**Identity condition:**

$$M M^{-1} = M \tilde{M} = 1$$

**Special case:** For $\theta = 0$, the rotor is the identity:

$$M(0) = \cos(0) - \sin(0) B = 1$$

## 2. Rotor Action on Points

### 2.1 Sandwich Product Transformation

A rotor transforms PGA points via the sandwich product:

$$\mathbf{p}' = M \mathbf{p} \tilde{M}$$

For a point $\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$, this produces:

$$\mathbf{p}' = \mathbf{e}_0 + (x')^m \mathbf{e}_m$$

where $(x')^m$ are the rotated Euclidean coordinates.

### 2.2 Rodrigues Formula in PGA

For rotation in plane $B$ by angle $\theta$, the coordinate transformation is:

$$x'^m = x^m + \sin(\theta) B^{mp} x_p + (1 - \cos(\theta)) B^{mp} B_p^{\ q} x_q$$

This is the Rodrigues rotation formula expressed through bivector components.

**3D rotation in $xy$-plane:** Using $B = \mathbf{e}_{12}$ (with $B^{12} = 1$):

$$
\begin{align}
	x' &= x \cos\theta - y \sin\theta \\ \\
	y' &= x \sin\theta + y \cos\theta \\ \\
	z' &= z
\end{align}
$$

**Specific angles:**

For $\theta = \pi/2$ (90° rotation):

$$
\begin{align}
	(1, 0, 0) &\mapsto (0, 1, 0) \\ \\
	(0, 1, 0) &\mapsto (-1, 0, 0) \\ \\
	(0, 0, 1) &\mapsto (0, 0, 1)
\end{align}
$$

For $\theta = \pi$ (180° rotation):

$$
\begin{align}
	(x, y, z) &\mapsto (-x, -y, z)
\end{align}
$$

### 2.3 Plane Invariance

The rotation plane $B$ is invariant under the rotor action:

$$M B \tilde{M} = B$$

All vectors orthogonal to $B$ rotate within the plane, while vectors in the orthogonal complement of $B$ remain fixed.

## 3. Translator Structure

### 3.1 Exponential Form

A translator in PGA is constructed from a displacement vector $\mathbf{t}$ via:

$$M = e^{-\mathbf{t}/2} = 1 - \frac{1}{2} t^m \mathbf{e}_{0m}$$

where $\mathbf{t} = t^m \mathbf{e}_{0m}$ is the degenerate bivector encoding translation.

**Multivector structure:**

$$M = M_0 \mathbf{1} + M^{0m} \mathbf{e}_{0m}$$

with:

$$M_0 = 1$$

$$M^{0m} = -\frac{1}{2} t^m$$

**Grade content:** A translator contains only grades {0, 2}, with the bivector part purely degenerate (involving $\mathbf{e}_0$).

### 3.2 Translator Action

A translator shifts points by the displacement vector:

$$\mathbf{p}' = M \mathbf{p} \tilde{M}$$

For point $\mathbf{p} = \mathbf{e}_0 + x^m \mathbf{e}_m$ translated by $t^m$:

$$\mathbf{p}' = \mathbf{e}_0 + (x^m + t^m) \mathbf{e}_m$$

**Coordinate transformation:**

$$x'^m = x^m + t^m$$

This is pure additive translation in Euclidean coordinates.

### 3.3 Translator Composition

Translators compose additively. For displacements $\mathbf{s}$ and $\mathbf{t}$:

$$M_{\mathbf{t}} M_{\mathbf{s}} = M_{\mathbf{t} + \mathbf{s}}$$

**Proof:** Expanding the exponentials:

$$
\begin{align}
	M_{\mathbf{t}} M_{\mathbf{s}} &= e^{-\mathbf{t}/2} e^{-\mathbf{s}/2} \\ \\
	&= \left(1 - \frac{\mathbf{t}}{2}\right)\left(1 - \frac{\mathbf{s}}{2}\right) \\ \\
	&= 1 - \frac{\mathbf{t} + \mathbf{s}}{2} + \frac{\mathbf{t} \mathbf{s}}{4} \\ \\
	&= 1 - \frac{\mathbf{t} + \mathbf{s}}{2}
\end{align}
$$

since $\mathbf{t} \mathbf{s} = 0$ (degenerate bivectors anti-commute to zero).

**Commutativity:** Translators commute:

$$M_{\mathbf{t}} M_{\mathbf{s}} = M_{\mathbf{s}} M_{\mathbf{t}}$$

### 3.4 Translator Versor Property

Translators satisfy the versor condition:

$$M \tilde{M} = 1$$

**Proof:** 

$$
\begin{align}
	M \tilde{M} &= \left(1 - \frac{\mathbf{t}}{2}\right)\left(1 + \frac{\mathbf{t}}{2}\right) \\ \\
	&= 1 - \frac{\mathbf{t}^2}{4} \\ \\
	&= 1
\end{align}
$$

since $\mathbf{t}^2 = 0$ for degenerate bivectors.

## 4. Rotation About Arbitrary Center

### 4.1 Decomposition Formula

Rotation about center $\mathbf{c}$ by angle $\theta$ in plane $B$ decomposes as:

$$M = T_{\mathbf{c}} R T_{-\mathbf{c}}$$

where:
- $T_{\mathbf{c}}$ translates by $\mathbf{c}$
- $R$ rotates about origin
- $T_{-\mathbf{c}}$ translates by $-\mathbf{c}$

**Explicit form:**

$$M = \left(1 - \frac{c^m \mathbf{e}_{0m}}{2}\right) \left(\cos(\theta/2) - \sin(\theta/2) B\right) \left(1 + \frac{c^m \mathbf{e}_{0m}}{2}\right)$$

### 4.2 Center Point Invariance

The center point is a fixed point of the transformation:

$$M \mathbf{p}_c \tilde{M} = \mathbf{p}_c$$

where $\mathbf{p}_c = \mathbf{e}_0 + c^m \mathbf{e}_m$.

**Proof:** By construction, translating to origin maps $\mathbf{p}_c \mapsto \mathbf{e}_0$, which is invariant under rotation about origin.

### 4.3 Line Exponential Equivalence

The decomposition formula is equivalent to the line exponential:

$$M = e^{-L\theta/2}$$

where $L = \mathbf{p}_c \wedge B$ is the Plücker line encoding both the rotation plane and center.

**Line structure:**

$$L = \mathbf{p}_c \wedge B = c^m B^{nq} \mathbf{e}_{0mnq} + B^{mn} \mathbf{e}_{0mn}$$

This is a grade-3 object in PGA (4D homogeneous space for 3D Euclidean geometry).

## 5. Motor Composition Algebra

### 5.1 Geometric Product Composition

Motors compose via the geometric product:

$$M_3 = M_2 M_1$$

Applied right-to-left:

$$M_3(\mathbf{p}) = M_2(M_1(\mathbf{p}))$$

**Non-commutativity:** In general, $M_2 M_1 \neq M_1 M_2$.

**Example:** Rotation then translation differs from translation then rotation.

### 5.2 Associativity

Motor composition is associative:

$$(M_3 M_2) M_1 = M_3 (M_2 M_1)$$

This follows from the associativity of the geometric product.

### 5.3 Identity and Inverses

**Identity motor:** $M_e = 1$ (grade-0 scalar).

**Inverse:** For any motor $M$:

$$M M^{-1} = M^{-1} M = 1$$

Since motors are versors:

$$M^{-1} = \tilde{M}$$

**Inverse transformation:**

$$M^{-1}(\mathbf{p}) = \tilde{M} \mathbf{p} M$$

### 5.4 Closure Under Composition

The set of motors is closed under composition:
- Rotor × Rotor = Rotor (composition of rotations)
- Translator × Translator = Translator (composition of translations)
- Rotor × Translator = Motor (general rigid transformation)

All results remain versors: $(M_2 M_1) \widetilde{(M_2 M_1)} = 1$.

## 6. Geometric Invariants

### 6.1 Distance Preservation (Isometry)

Motors preserve Euclidean distances. For any two points $\mathbf{p}_1, \mathbf{p}_2$:

$$d(\mathbf{p}'_1, \mathbf{p}'_2) = d(\mathbf{p}_1, \mathbf{p}_2)$$

where $\mathbf{p}'_i = M \mathbf{p}_i \tilde{M}$ and $d$ is the Euclidean distance.

**Proof:** The sandwich product preserves the PGA inner product, which induces the Euclidean metric on affine space.

### 6.2 Angle Preservation

Motors preserve angles between vectors. For vectors $\mathbf{u}, \mathbf{v}$:

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}||\mathbf{v}|} = \frac{\mathbf{u}' \cdot \mathbf{v}'}{|\mathbf{u}'||\mathbf{v}'|}$$

where $\mathbf{u}' = M \mathbf{u} \tilde{M}$ and similarly for $\mathbf{v}$.

### 6.3 Orientation Preservation

Motors preserve orientation. The determinant of the induced linear transformation on Euclidean space is:

$$\det(M) = +1$$

Motors form the **special Euclidean group** $SE(d)$, consisting of orientation-preserving isometries.

### 6.4 Rigid Body Transformations

Motors represent the complete group of rigid body transformations:
- Preserve distances (isometry)
- Preserve angles (conformal)
- Preserve orientation (proper)
- Form a Lie group under composition

## 7. Special Cases and Identities

### 7.1 Rotation-Translation Commutator

Rotations and translations do not commute. For rotor $R$ and translator $T$:

$$R T \neq T R$$

**Commutator:**

$$[R, T] = R T R^{-1} T^{-1} \neq 1$$

This measures the difference between "translate then rotate" and "rotate then translate".

### 7.2 Half-Angle Formulation

The exponential uses half-angles:

$$M = e^{-B\theta/2}$$

This ensures the sandwich product gives the full angle:

$$M \mathbf{v} \tilde{M} \text{ rotates by angle } \theta$$

**Why half-angle?** The sandwich product applies the transformation twice (once on each side), so $\theta/2$ compounds to $\theta$.

### 7.3 Plane Rotations in Higher Dimensions

In $d$-dimensional space, there are $\binom{d}{2}$ independent rotation planes.

**4D example:** Six independent bivectors:
- $\mathbf{e}_{12}, \mathbf{e}_{13}, \mathbf{e}_{14}$ (three planes involving $x$)
- $\mathbf{e}_{23}, \mathbf{e}_{24}$ (two planes involving $y$)
- $\mathbf{e}_{34}$ (one plane for $zw$)

Each bivector generates an independent rotation in its plane.

### 7.4 Euclidean vs Degenerate Components

A general motor has both Euclidean and degenerate bivector components:

$$M = M_0 \mathbf{1} + M^{mn} \mathbf{e}_{mn} + M^{0m} \mathbf{e}_{0m}$$

where:
- $M^{mn}$ (Euclidean, $m,n \geq 1$): encodes rotation
- $M^{0m}$ (degenerate): encodes translation

**Decomposition theorem:** Any motor can be uniquely decomposed as:

$$M = T \cdot R = \text{translator} \times \text{rotor}$$

## 8. Collection Dimensions and Broadcasting

### 8.1 Parameterized Families

A single bivector $B$ with varying angle $\theta_k$ produces a family of rotors:

$$M_k(\theta_k) = \cos(\theta_k/2) - \sin(\theta_k/2) B$$

This creates a collection dimension over the angle parameter.

### 8.2 Simultaneous Transformations

Multiple motors can act on multiple points simultaneously through einsum broadcasting:

$$\mathbf{p}'_{ij} = M_i \mathbf{p}_j \tilde{M}_i$$

where $i$ indexes motors and $j$ indexes points.

**Result shape:** Collection dimensions broadcast according to standard numpy rules.

### 8.3 Animation and Interpolation

Linear interpolation of angles produces smooth animation:

$$M(t) = \cos(t\theta/2) - \sin(t\theta/2) B$$

for $t \in [0, 1]$, smoothly interpolating from identity to full rotation.

**Velocity:** The derivative gives the angular velocity:

$$\frac{dM}{dt} = -\frac{\theta B}{2} M$$

## 9. Higher-Dimensional Properties

### 9.1 Dimensionality Scaling

In $d$-dimensional Euclidean space, PGA uses $(d+1)$-dimensional homogeneous coordinates.

**Component counts:**
- Grade-0: 1 component (scalar)
- Grade-2: $\binom{d+1}{2}$ components (bivector)
- Total motor components: $1 + \binom{d+1}{2}$

**3D example:** $1 + \binom{4}{2} = 1 + 6 = 7$ components
**4D example:** $1 + \binom{5}{2} = 1 + 10 = 11$ components

### 9.2 Dual Representations

The Hodge dual establishes correspondences in specific dimensions:

**3D only:** Bivectors dual to vectors via $\star : \mathbf{e}_{mn} \leftrightarrow \mathbf{e}_p$

This correspondence fails in other dimensions, which is why axis-angle representation only works in 3D.

**General principle:** Rotations always occur in planes (bivectors), never around axes (vectors).

