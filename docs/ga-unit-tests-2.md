# Geometric Product and Transformations - Unit Tests

This document specifies unit tests for the geometric product operations and rigid transformations (rotors, motors, versors). Tests are organized by operation, with mathematical properties and specific test cases.

## Geometric Product

### Basic Properties

**Associativity**
For blades $\mathbf{u}$, $\mathbf{v}$, $\mathbf{w}$:

$$
(\mathbf{u}\mathbf{v})\mathbf{w} = \mathbf{u}(\mathbf{v}\mathbf{w})
$$

Test in 2D, 3D, 4D with various grade combinations.

**Distributivity**
For blades $\mathbf{u}$, $\mathbf{v}$, $\mathbf{w}$:

$$
\mathbf{u}(\mathbf{v} + \mathbf{w}) = \mathbf{u}\mathbf{v} + \mathbf{u}\mathbf{w}
$$

**Vector Contraction Law**
For any vector $\mathbf{v}$:

$$
\mathbf{v}^2 = \mathbf{v} \cdot \mathbf{v} = |\mathbf{v}|^2
$$

Test with unit vectors (should give 1) and arbitrary vectors.

**Anticommutativity for Orthogonal Vectors**
For orthogonal vectors $\mathbf{u} \perp \mathbf{v}$:

$$
\mathbf{u}\mathbf{v} = -\mathbf{v}\mathbf{u}
$$

Test in 3D with $\mathbf{e}_1, \mathbf{e}_2$.

**Commutativity for Parallel Vectors**
For parallel vectors $\mathbf{u} \parallel \mathbf{v}$:

$$
\mathbf{u}\mathbf{v} = \mathbf{v}\mathbf{u} = \mathbf{u} \cdot \mathbf{v}
$$

Test with $\mathbf{v} = 2\mathbf{u}$.

### Vector × Vector Product

**Grade Decomposition**
For vectors $\mathbf{u}$, $\mathbf{v}$:

$$
\mathbf{u}\mathbf{v} = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \wedge \mathbf{v}
$$

Components:
- Grade 0: $\braket{\mathbf{u}\mathbf{v}}_0 = \mathbf{u} \cdot \mathbf{v}$
- Grade 2: $\braket{\mathbf{u}\mathbf{v}}_2 = \mathbf{u} \wedge \mathbf{v}$

**Test Case: 2D Orthogonal Vectors**
- $\mathbf{u} = [1, 0]$, $\mathbf{v} = [0, 1]$
- Expected: grade-0 = 0, grade-2 = $\mathbf{e}_{12}$ (single component = 1)

**Test Case: 3D Perpendicular**
- $\mathbf{u} = [1, 0, 0]$, $\mathbf{v} = [0, 1, 0]$
- Expected: grade-0 = 0, grade-2 = $\mathbf{e}_{12}$ with component $B^{12} = 1$

**Test Case: 3D Arbitrary Angle**
- $\mathbf{u} = [1, 0, 0]$, $\mathbf{v} = [\cos\theta, \sin\theta, 0]$
- Expected: grade-0 = $\cos\theta$, grade-2 has magnitude $\sin\theta$

### Vector × Bivector Product

**Grade Decomposition**
For vector $\mathbf{u}$ (grade 1) and bivector $\mathbf{B}$ (grade 2):

$$
\mathbf{u}\mathbf{B} = \mathbf{u} \lrcorner \mathbf{B} + \mathbf{u} \wedge \mathbf{B}
$$

Components:
- Grade 1: interior product (contraction)
- Grade 3: exterior product

**Test Case: 3D - Vector in Bivector Plane**
- $\mathbf{u} = [1, 0, 0]$, $\mathbf{B} = \mathbf{e}_{12}$ (the $xy$-plane)
- Expected: grade-1 component (contraction exists), grade-3 = 0 (can't wedge to make trivector from coplanar elements)

**Test Case: 3D - Vector Perpendicular to Bivector**
- $\mathbf{u} = [0, 0, 1]$, $\mathbf{B} = \mathbf{e}_{12}$
- Expected: grade-1 = 0 (no contraction), grade-3 = $\mathbf{e}_{123}$ (full trivector)

### Bivector × Bivector Product

**Grade Decomposition**
For bivectors $\mathbf{B}_1$, $\mathbf{B}_2$ (grade 2):

$$
\mathbf{B}_1\mathbf{B}_2 = \braket{\mathbf{B}_1\mathbf{B}_2}_0 + \braket{\mathbf{B}_1\mathbf{B}_2}_2 + \braket{\mathbf{B}_1\mathbf{B}_2}_4
$$

**Test Case: 3D - Orthogonal Bivectors**
In 3D, only grades 0 and 2 exist (no grade-4).
- $\mathbf{B}_1 = \mathbf{e}_{12}$, $\mathbf{B}_2 = \mathbf{e}_{23}$
- Expected: grade-0 (scalar overlap), grade-2 (commutator part)

**Test Case: 4D - Completely Orthogonal Bivectors**
- $\mathbf{B}_1 = \mathbf{e}_{12}$, $\mathbf{B}_2 = \mathbf{e}_{34}$
- Expected: grade-0 = 0, grade-2 = 0, grade-4 = $\mathbf{e}_{1234}$ (pseudoscalar)
- Verify: $\mathbf{B}_1\mathbf{B}_2 = \mathbf{B}_2\mathbf{B}_1$ (commute because orthogonal)

**Test Case: 3D - Unit Bivector Squares**
For unit bivectors in 3D:

$$
\mathbf{e}_{12}^2 = \mathbf{e}_{23}^2 = \mathbf{e}_{31}^2 = -1
$$

Test each and verify scalar result is $-1$.

### Dimension Scaling

**Test Case: 2D Pseudoscalar**
- $\mathbb{1}_{2D} = \mathbf{e}_{12}$
- Verify: $\mathbb{1}^2 = -1$ (behaves like imaginary unit)

**Test Case: 3D Pseudoscalar**
- $\mathbb{1}_{3D} = \mathbf{e}_{123}$
- Verify: $\mathbb{1}^2 = -1$

**Test Case: 4D Pseudoscalar**
- $\mathbb{1}_{4D} = \mathbf{e}_{1234}$
- Verify: $\mathbb{1}^2 = +1$ (sign change in even dimensions)

**General Pattern**
For $d$-dimensional Euclidean space:

$$
\mathbb{1}^2 = (-1)^{d(d-1)/2}
$$

Test for dimensions 2 through 6.

## Reversion and Inverse

### Reversion Properties

**Sign Pattern**
For grade-$k$ blade:

$$
\widetilde{\mathbf{A}} = (-1)^{k(k-1)/2} \mathbf{A}
$$

Test pattern:
- Grade 0, 1: $+$ (unchanged)
- Grade 2, 3: $-$ (sign flip)
- Grade 4, 5: $+$ (unchanged)
- Grade 6, 7: $-$ (sign flip)

**Reverse of Product**

$$
\widetilde{\mathbf{A}\mathbf{B}} = \widetilde{\mathbf{B}} \, \widetilde{\mathbf{A}}
$$

Test with vectors, bivectors, and mixed products.

**Involution Property**

$$
\widetilde{\widetilde{\mathbf{A}}} = \mathbf{A}
$$

Apply reverse twice, should get original blade.

### Inverse Properties

**Left and Right Inverse**
For blade $\mathbf{u}$ with inverse $\mathbf{u}^{-1}$:

$$
\mathbf{u}^{-1}\mathbf{u} = \mathbf{u}\mathbf{u}^{-1} = 1
$$

**Vector Inverse Formula**
For vector $\mathbf{v}$:

$$
\mathbf{v}^{-1} = \frac{\mathbf{v}}{|\mathbf{v}|^2}
$$

Test with $\mathbf{v} = [3, 4]$ in 2D:
- $|\mathbf{v}|^2 = 25$
- $\mathbf{v}^{-1} = [0.12, 0.16]$
- Verify: $\mathbf{v}^{-1}\mathbf{v} = 1$

**Bivector Inverse**
For bivector $\mathbf{B}$ with $\mathbf{B}^2 = -|\mathbf{B}|^2$:

$$
\mathbf{B}^{-1} = \frac{\widetilde{\mathbf{B}}}{\mathbf{B}\widetilde{\mathbf{B}}}
$$

Test with unit bivector $\mathbf{e}_{12}$ in 3D:
- $\mathbf{e}_{12}^2 = -1$
- $\mathbf{e}_{12}^{-1} = -\mathbf{e}_{12}$

## Reflections

### Basic Reflection

**Reflection Formula**
For vector $\mathbf{v}$ reflected through hyperplane with unit normal $\mathbf{n}$:

$$
\mathbf{v}' = -\mathbf{n}\mathbf{v}\mathbf{n}
$$

**Test Case: 2D Reflection Across x-axis**
- $\mathbf{v} = [3, 4]$, $\mathbf{n} = [0, 1]$ (normal to x-axis)
- Expected: $\mathbf{v}' = [3, -4]$

**Test Case: 3D Reflection Through xy-plane**
- $\mathbf{v} = [1, 2, 3]$, $\mathbf{n} = [0, 0, 1]$
- Expected: $\mathbf{v}' = [1, 2, -3]$

**Test Case: 3D Reflection Through Arbitrary Plane**
- $\mathbf{v} = [1, 0, 0]$, $\mathbf{n} = [1/\sqrt{2}, 1/\sqrt{2}, 0]$
- Expected: $\mathbf{v}' = [0, 1, 0]$ (45° reflection)

### Reflection Properties

**Involution**
Reflecting twice through same plane returns original:

$$
-\mathbf{n}(-\mathbf{n}\mathbf{v}\mathbf{n})\mathbf{n} = \mathbf{v}
$$

**Preserves Magnitude**

$$
|\mathbf{v}'| = |\mathbf{v}|
$$

**Orthogonal Components**
- Component parallel to $\mathbf{n}$: flips sign
- Component perpendicular to $\mathbf{n}$: unchanged

Test by decomposing $\mathbf{v} = \mathbf{v}_\parallel + \mathbf{v}_\perp$ and verifying separately.

## Rotors

### Rotor Construction

**From Bivector and Angle**
For unit bivector $\mathbf{B}$ with $\mathbf{B}^2 = -1$:

$$
R = e^{-\theta\mathbf{B}/2} = \cos\frac{\theta}{2} - \mathbf{B}\sin\frac{\theta}{2}
$$

**Test Case: 2D Rotation**
- $\mathbf{B} = \mathbf{e}_{12}$, $\theta = \pi/2$ (90° rotation)
- Expected: $R = \cos(\pi/4) - \mathbf{e}_{12}\sin(\pi/4) = \frac{1}{\sqrt{2}}(1 - \mathbf{e}_{12})$

**Test Case: 3D Rotation About z-axis**
- $\mathbf{B} = -\mathbf{e}_{12}$ (plane perpendicular to z)
- $\theta = \pi/2$
- Expected: $R = \frac{1}{\sqrt{2}}(1 + \mathbf{e}_{12})$

**From Two Vectors**
Rotor rotating $\mathbf{u}$ to $\mathbf{v}$:

$$
R = \frac{\mathbf{v}\mathbf{u} + |\mathbf{v}||\mathbf{u}|}{|\mathbf{v}\mathbf{u} + |\mathbf{v}||\mathbf{u}||}
$$

**Test Case: 2D - 90° Rotation**
- $\mathbf{u} = [1, 0]$, $\mathbf{v} = [0, 1]$
- Construct rotor, verify it rotates $\mathbf{u}$ to $\mathbf{v}$

**Test Case: 3D - Align Vectors**
- $\mathbf{u} = [1, 0, 0]$, $\mathbf{v} = [1/\sqrt{2}, 1/\sqrt{2}, 0]$
- Construct rotor, apply to $\mathbf{u}$, should get $\mathbf{v}$

### Rotor Properties

**Unit Magnitude**

$$
R\widetilde{R} = 1
$$

Test that $|R| = 1$ for all constructed rotors.

**Even Grade**
Rotors contain only even grades (0, 2, 4, ...).

Verify component structure for rotors in 2D (grades 0,2), 3D (grades 0,2), 4D (grades 0,2,4).

**Composition**
For rotors $R_1$ and $R_2$:

$$
R = R_2 R_1
$$

applies $R_1$ first, then $R_2$.

**Test Case: Two 90° Rotations in 2D**
- $R_1$: rotate 90° counterclockwise
- $R_2$: rotate 90° counterclockwise
- $R = R_2 R_1$: should be 180° rotation
- Verify: $R = -1$ (scalar -1 represents 180° rotation in 2D)

**Test Case: Commuting Rotations in 4D**
- $R_1$: rotation in $\mathbf{e}_{12}$ plane
- $R_2$: rotation in $\mathbf{e}_{34}$ plane
- Verify: $R_1 R_2 = R_2 R_1$ (orthogonal planes commute)

### Rotor Application

**Sandwich Product**

$$
\mathbf{v}' = R\mathbf{v}\widetilde{R}
$$

**Test Case: 2D - 90° Rotation**
- $R = \frac{1}{\sqrt{2}}(1 - \mathbf{e}_{12})$
- $\mathbf{v} = [1, 0]$
- Expected: $\mathbf{v}' = [0, 1]$

**Test Case: 3D - 180° About x-axis**
- $R = \mathbf{e}_{23}$ (bivector alone = 180° rotation)
- $\mathbf{v} = [0, 1, 0]$
- Expected: $\mathbf{v}' = [0, -1, 0]$

**Preserves Grade and Magnitude**
- Grade: $\text{grade}(\mathbf{v}') = \text{grade}(\mathbf{v})$
- Magnitude: $|\mathbf{v}'| = |\mathbf{v}|$

Test with vectors, bivectors, trivectors.

**Leaves Perpendicular Components Fixed**
For rotor $R = \cos(\theta/2) - \mathbf{B}\sin(\theta/2)$, vectors perpendicular to plane $\mathbf{B}$ are unchanged.

**Test Case: 3D Rotation in xy-plane**
- $R$: rotation in $\mathbf{e}_{12}$ plane (about z-axis)
- $\mathbf{v} = [0, 0, 1]$ (perpendicular to rotation plane)
- Expected: $\mathbf{v}' = \mathbf{v}$ (unchanged)

### Parameter Extraction

**Angle Extraction**

$$
\theta = 2\arccos(\braket{R}_0)
$$

Test by constructing rotor with known angle, extracting, comparing.

**Plane Extraction**
For simple rotor:

$$
\mathbf{B} = \frac{\braket{R}_2}{|\braket{R}_2|}
$$

Test by constructing rotor from known bivector, extracting, comparing.

**Matrix Conversion**
Rotor should produce same rotation matrix as classical formulas.

**Test Case: 3D Rodrigues Formula**
- Construct rotor for rotation about axis $\mathbf{n}$ by angle $\theta$
- Convert to matrix
- Compare with Rodrigues formula:

$$
A_{mn} = \cos\theta \, \delta_{mn} + (1-\cos\theta) n_m n_n + \sin\theta \, \varepsilon_{mnp} n^p
$$

### Rotor Interpolation

**SLERP**
For $t \in [0, 1]$:

$$
R(t) = R_0(R_0^{-1}R_1)^t
$$

**Test Case: 2D - Interpolate Between 0° and 90°**
- $R_0 = 1$ (identity)
- $R_1 = \frac{1}{\sqrt{2}}(1 - \mathbf{e}_{12})$ (90° rotation)
- $R(0.5)$: should be 45° rotation
- Verify by applying to $\mathbf{v} = [1, 0]$, expect $[\cos 45°, \sin 45°]$

**Smoothness**
Path $R(t)$ should be smooth (continuous derivatives).

Test by computing rotor for several $t$ values, verify gradual change.

## Translators and Motors (PGA)

### Translator Construction

**From Displacement**
For Euclidean displacement $\mathbf{d}$:

$$
T = 1 + \frac{1}{2}\mathbf{e}_0 \wedge \mathbf{d}
$$

**Test Case: 3D Translation**
- Displacement: $\mathbf{d} = [1, 0, 0]$ (translate 1 unit in x)
- Construct translator
- Apply to point $\mathbf{p} = \mathbf{e}_0 + 0\mathbf{e}_1 + 0\mathbf{e}_2 + 0\mathbf{e}_3$ (origin)
- Expected: $\mathbf{p}' = \mathbf{e}_0 + 1\mathbf{e}_1 + 0\mathbf{e}_2 + 0\mathbf{e}_3$

**Test Case: 2D Translation**
- Displacement: $\mathbf{d} = [3, 4]$
- Point: origin
- Expected: point at $(3, 4)$

### Translator Properties

**Composition**

$$
T = T_2 T_1
$$

applies translation $\mathbf{d}_1$ then $\mathbf{d}_2$.

**Test Case: Two Translations**
- $T_1$: translate by $[1, 0, 0]$
- $T_2$: translate by $[0, 2, 0]$
- $T = T_2 T_1$: should translate by $[1, 2, 0]$

**Inverse**

$$
T^{-1} = 1 - \frac{1}{2}\mathbf{e}_0 \wedge \mathbf{d}
$$

Test: $T^{-1}T = 1$

### Motor Construction

**From Rotor and Translator**

$$
M = RT
$$

combines rotation $R$ followed by translation $T$.

**Test Case: 2D - Rotate Then Translate**
- $R$: 90° rotation
- $T$: translate by $[1, 0]$
- Apply to point at origin
- Expected: point at $(1, 0)$ (rotates origin, which stays at origin, then translates)

**Test Case: 2D - Rotate About Point**
To rotate by $\theta$ about point $\mathbf{c}$:
1. Translate by $-\mathbf{c}$
2. Rotate by $\theta$
3. Translate by $+\mathbf{c}$

Construct motor, test by rotating square about its center.

### Motor Properties

**Screw Motion**
Motor can represent combined rotation + translation along rotation axis.

**Test Case: 3D Screw**
- Rotation: 90° about z-axis
- Translation: 1 unit along z-axis
- Point at $(1, 0, 0)$
- Expected: point at $(0, 1, 1)$

**Decomposition**
Given motor $M$, extract rotation and translation parts.

Test by constructing motor from known $R$ and $T$, decomposing, comparing.

## Versors

### Versor Construction

**From Reflections**

$$
V = \mathbf{n}_k \cdots \mathbf{n}_2 \mathbf{n}_1
$$

**Test Case: Two Reflections = Rotation**
- $\mathbf{n}_1 = [1, 0]$ (normal to x-axis)
- $\mathbf{n}_2 = [\cos\theta, \sin\theta]$
- Product: rotation by $2\theta$
- Verify equals rotor for angle $2\theta$

**Test Case: Three Reflections = Rotoreflection**
- Odd number of reflections
- Verify determinant = -1 (orientation reversal)

### Versor Properties

**Unit Magnitude**
All versors satisfy:

$$
V\widetilde{V} = \pm 1
$$

**Cartan-Dieudonné Theorem**
Any orthogonal transformation in $d$ dimensions can be written as product of at most $d$ reflections.

**Test Case: 3D Rotation from At Most 3 Reflections**
Given arbitrary 3D rotation matrix, factor into reflections, verify product gives same rotation.

## Edge Cases and Numerical Stability

### Near-Zero Cases

**Small Angle Rotations**
For $\theta \approx 0$:

$$
R \approx 1 - \frac{\theta}{2}\mathbf{B}
$$

Test with $\theta = 10^{-8}$, verify rotor is nearly identity.

**Nearly Parallel Vectors**
When constructing rotor from $\mathbf{u}$ to $\mathbf{v}$ with $\mathbf{u} \approx \mathbf{v}$:
- Should give nearly identity rotor
- Test with $\mathbf{v} = \mathbf{u} + 10^{-6}\mathbf{w}$

**Nearly Antiparallel Vectors**
When $\mathbf{v} \approx -\mathbf{u}$:
- Rotation is nearly 180°
- Any perpendicular axis works
- Test numerical stability

### High Dimensions

**4D Double Rotation**
Test simultaneous independent rotations in orthogonal planes.

**5D and Beyond**
Verify dimensional scaling:
- Number of rotation planes: $\binom{d}{2}$
- Rotor components: grades 0, 2, 4, ... up to $d$ or $d-1$

### Batch Operations

**Vectorized Rotations**
Apply same rotor to batch of vectors:
- Input: shape $(N, 3)$
- Output: shape $(N, 3)$

**Batch Rotors**
Different rotor for each vector:
- Rotors: shape $(N, \text{components})$
- Vectors: shape $(N, 3)$
- Verify broadcasting works correctly

## Integration Tests

### Round-Trip Tests

**Rotor → Matrix → Rotor**
- Construct rotor
- Convert to matrix
- Convert back to rotor
- Should match original (up to sign ambiguity)

**Angle-Plane → Rotor → Angle-Plane**
- Start with bivector $\mathbf{B}$ and angle $\theta$
- Construct rotor
- Extract angle and plane
- Should match original

### Composition Chains

**Multiple Rotations**
Chain 5-10 rotations, verify:
- Final rotor is unit
- Applied transformation matches matrix product

**Mixed Transformations (PGA)**
Chain rotations and translations:
- Verify motor composition
- Test on multiple points
- Compare with matrix/vector approach

### Physical Validations

**Conservation Laws**
- Magnitude preservation: $|\mathbf{v}'| = |\mathbf{v}|$
- Angle preservation: $\mathbf{u}' \cdot \mathbf{v}' = \mathbf{u} \cdot \mathbf{v}$
- Volume preservation: $(\mathbf{u} \wedge \mathbf{v})' = \mathbf{u}' \wedge \mathbf{v}'$ (same magnitude)

**Determinant**
- Proper rotations: $\det = +1$
- Improper rotations (with reflection): $\det = -1$

Test by converting to matrix and checking determinant.
