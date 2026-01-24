# Geometric Algebra Foundation: Missing Core Operations

## Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| **Phase 1: Exponentials** | | |
| `exp_blade()` | Implemented | `operations/exponential.py` |
| `log_versor()` | Implemented | `operations/exponential.py` |
| `slerp()` | Implemented | `operations/exponential.py` |
| `rotor()` refactored | Implemented | `transforms/rotations.py` |
| **Phase 2: Matrix Interfaces** | | |
| `blade_to_vector()` | Implemented | `operations/matrix_rep.py` |
| `vector_to_blade()` | Implemented | `operations/matrix_rep.py` |
| `multivector_to_vector()` | Implemented | `operations/matrix_rep.py` |
| `vector_to_multivector()` | Implemented | `operations/matrix_rep.py` |
| `left_matrix()` | Implemented | `operations/matrix_rep.py` |
| `right_matrix()` | Implemented | `operations/matrix_rep.py` |
| `operator_to_matrix()` | Implemented | `operations/matrix_rep.py` |
| **Phase 3: Spectral Tools** | | |
| `bivector_to_skew_matrix()` | Implemented | `operations/spectral.py` |
| `bivector_eigendecomposition()` | Implemented | `operations/spectral.py` |
| `blade_principal_vectors()` | Implemented | `operations/spectral.py` |
| **Phase 4: Advanced Linear Algebra** | | |
| Geometric regularization | Not implemented | - |
| Iterative solvers | Not implemented | - |

---

## Overview

This document outlines two critical categories of operations currently absent from the morphis geometric algebra library but essential for a complete foundation: **exponentials and logarithms** for blades/multivectors, and **enhanced linear algebra integration** preserving geometric structure. These additions would significantly expand the library's capabilities for both theoretical geometric algebra and practical applications in electromagnetic field analysis and inverse problems.

The morphis library currently provides a solid foundation with wedge products, geometric products, duality operations, and basic transformations. However, the absence of exponential maps and structure-preserving linear operators represents a fundamental gap between the *algebra* (infinitesimal transformations) and the *group* (finite transformations), and between geometric operations and numerical linear algebra.

---

## Part I: Exponentials and Logarithms

### Mathematical Foundation

The exponential map is the bridge connecting Lie algebras to Lie groups in geometric algebra. For a blade $\mathbf{B}$ where $\mathbf{B}^2 = \lambda$ is scalar, the exponential follows from the Taylor series:

$$
e^{\mathbf{B}} = 1 + \mathbf{B} + \frac{\mathbf{B}^2}{2!} + \frac{\mathbf{B}^3}{3!} + \cdots
$$

This series telescopes based on the sign of $\mathbf{B}^2$:

$$
e^{\mathbf{B}} = \begin{cases}
\cosh\sqrt{\lambda} + \frac{\mathbf{B}}{\sqrt{\lambda}} \sinh\sqrt{\lambda} & \text{if } \lambda > 0 \text{ (hyperbolic)} \\
\cos\sqrt{-\lambda} + \frac{\mathbf{B}}{\sqrt{-\lambda}} \sin\sqrt{-\lambda} & \text{if } \lambda < 0 \text{ (trigonometric)} \\
1 + \mathbf{B} & \text{if } \lambda = 0 \text{ (nilpotent)}
\end{cases}
$$

The logarithm is the inverse operation, extracting the generator from a versor. For a rotor $\mathbf{R} = a + \mathbf{B}$ where $a$ is scalar and $\mathbf{B}$ is bivector:

$$
\log \mathbf{R} = \arctan2(|\mathbf{B}|, a) \cdot \frac{\mathbf{B}}{|\mathbf{B}|}
$$

### Computational Efficiency

A critical advantage of blade exponentials is computational simplicity. Unlike matrix exponentials which require Padé approximation or eigendecomposition, blade exponentials reduce to closed-form scalar operations.

The Taylor series naturally separates into scalar and blade components. Since $\mathbf{B}^2 = \lambda$ is scalar, all higher powers alternate: $\mathbf{B}^3 = \lambda \mathbf{B}$, $\mathbf{B}^4 = \lambda^2$, $\mathbf{B}^5 = \lambda^2 \mathbf{B}$, and so on. Regrouping the series:

$$
\begin{align}
e^{\mathbf{B}} &= \left(1 + \frac{\mathbf{B}^2}{2!} + \frac{\mathbf{B}^4}{4!} + \cdots\right) + \mathbf{B}\left(1 + \frac{\mathbf{B}^2}{3!} + \frac{\mathbf{B}^4}{5!} + \cdots\right) \\ \\
&= \underbrace{\left(1 + \frac{\lambda}{2!} + \frac{\lambda^2}{4!} + \cdots\right)}_{\text{scalar series}} + \mathbf{B} \, \underbrace{\left(1 + \frac{\lambda}{3!} + \frac{\lambda^2}{5!} + \cdots\right)}_{\text{scalar series}}
\end{align}
$$

These scalar series are simply the Taylor expansions of $\cos\sqrt{-\lambda}$ and $\frac{\sin\sqrt{-\lambda}}{\sqrt{-\lambda}}$ (for $\lambda < 0$) or their hyperbolic counterparts (for $\lambda > 0$). No infinite summation is ever performed—the computation evaluates closed-form expressions.

The actual execution requires:
1. One geometric product to compute $\mathbf{B}^2$ using existing einsum operations
2. Extract the scalar value $\lambda$ via grade projection
3. Compute scalar square root and trigonometric/hyperbolic functions
4. Scale the blade by scalar coefficients

For a bivector in Euclidean 3D space where $\mathbf{B}^2 = -|\mathbf{B}|^2$, this amounts to computing $\cos(|\mathbf{B}|)$ and $\sin(|\mathbf{B}|)/|\mathbf{B}|$ as pure scalars, then forming $e^{\mathbf{B}} = \cos(|\mathbf{B}|) + \frac{\sin(|\mathbf{B}|)}{|\mathbf{B}|} \mathbf{B}$.

This approach is **more efficient** than matrix exponentials. For $n \times n$ matrices, standard algorithms require $O(n^3)$ operations with iterative refinement. Blade exponentials require one einsum operation (already optimized in the existing `products.py` module) followed by scalar arithmetic. Moreover, when applied to collections of blades, NumPy's broadcasting handles all scalar operations vectorially—computing exponentials for thousands of bivectors at different spatial points costs essentially the same as computing one.

The mathematical guarantee enabling this efficiency is that for simple $k$-blades, $\mathbf{B}^2$ is always scalar: $\mathbf{B}^2 = (-1)^{k(k-1)/2} |\mathbf{B}|^2$. The existing geometric product implementation correctly extracts all grade components; blade exponentials simply use the grade-0 part.

### Current State in Morphis

The library currently implements:
- **Geometric product** (`operations/products.py`): Full implementation including blade-blade and multivector-multivector products with systematic einsum operations
- **Reversion** (`operations/products.py`): The anti-involution $\tilde{\mathbf{A}} = (-1)^{k(k-1)/2} \mathbf{A}$ for grade-$k$ blades
- **Inverse** (`operations/products.py`): Computes $\mathbf{A}^{-1} = \tilde{\mathbf{A}}/(\mathbf{A}\tilde{\mathbf{A}})$ when the denominator is scalar
- **Sandwich product infrastructure**: The `actions.py` module applies transformations via $\mathbf{M} \mathbf{x} \tilde{\mathbf{M}}$

What's **missing**:
- Exponential map taking blades to versors
- Logarithm map $\log: \text{Spin}(V) \to \Lambda^k(V)$ extracting generators
- Rotor construction from bivectors beyond explicit formulas
- Interpolation between rotors (slerp)

### Why This Matters

**1. Generating Rotors from Bivectors**

Currently, the library would need to construct rotors through explicit formulas or iterative methods. With exponentials, rotor generation becomes direct and mathematically natural:

A bivector $\mathbf{B}$ encodes both the rotation plane (its attitude) and magnitude (rotation angle). The rotor is simply:

$$
\mathbf{R} = e^{-\mathbf{B}/2}
$$

The half-angle appears because the sandwich product $\mathbf{R} \mathbf{v} \tilde{\mathbf{R}}$ effectively applies the transformation twice. This is the fundamental construction missing from the current implementation.

**2. Interpolation Between Orientations**

The library's visualization modules (`animate_3d.py`, `animate_4d.py`) currently lack smooth orientation interpolation. The correct way to interpolate between rotors $\mathbf{R}_0$ and $\mathbf{R}_1$ is:

$$
\mathbf{R}(t) = \mathbf{R}_0 \, e^{t \log(\mathbf{R}_0^{-1} \mathbf{R}_1)}
$$

This spherical linear interpolation (slerp) maintains the group structure and produces constant angular velocity. Linear interpolation in rotor components, by contrast, causes acceleration artifacts and doesn't preserve unit magnitude.

**3. Differential Geometry Foundation**

Exponentials connect tangent spaces to manifolds. For the rotation group $\text{SO}(n)$, the tangent space at the identity is the Lie algebra of bivectors. The exponential map realizes this connection explicitly, taking bivectors (infinitesimal rotations) to rotors (finite rotations). This is the mathematical foundation needed for the currently empty `manifold/` directory, enabling flows, geodesics, and parallel transport.

**4. Composition via Addition**

An elegant property of the exponential map is that rotor composition corresponds (approximately) to bivector addition:

$$
e^{\mathbf{B}_1} e^{\mathbf{B}_2} \approx e^{\mathbf{B}_1 + \mathbf{B}_2}
$$

Exact when $[\mathbf{B}_1, \mathbf{B}_2] = 0$ (commuting bivectors); otherwise governed by the Baker-Campbell-Hausdorff formula. This provides an additive parameterization of the rotation group, crucial for optimization and control applications.

### Implementation Strategy

The core computational challenge is evaluating $\mathbf{B}^2$ and distinguishing the three cases. The existing `operations/products.py` module already computes geometric products via einsum. The exponential function would:

1. Compute $\mathbf{B}^2$ using the existing `geometric()` function
2. Extract the scalar component to determine $\lambda$
3. Branch based on $\text{sign}(\lambda)$
4. Apply the appropriate formula using NumPy trigonometric/hyperbolic functions

The logarithm is more subtle, requiring careful handling of:
- Branch cuts (periodicity of trigonometric functions)
- Numerical stability near the identity
- Multiple bivector generators producing the same rotor

For rotors $\mathbf{R} = a + \mathbf{B}$ where $a = \langle \mathbf{R} \rangle_0$ and $\mathbf{B} = \langle \mathbf{R} \rangle_2$:

1. Extract scalar and bivector components using existing grade projection
2. Compute bivector magnitude $|\mathbf{B}|$ using existing `norms.py` operations
3. Use `atan2(|B|, a)` for proper quadrant handling
4. Scale the normalized bivector by the angle

The existing `operations/norms.py` module provides `norm()` and `norm_squared()` functions that would serve as building blocks.

### Metric Signature Dependence

The sign of $\mathbf{B}^2$ depends critically on the metric signature:

- **Euclidean metrics** (all positive): Bivectors square to negative $\implies$ trigonometric exponentials
- **Minkowski metrics** ($(-,+,+,+)$ or $(+,-,-,-)$): Some bivectors square positive (boosts), others negative (rotations)
- **Degenerate metrics** (PGA): Some bivectors square to zero (null rotations/translations)

The implementation must query the metric to determine the behavior. The existing `Metric` class in `elements/metric.py` provides `signature` and `signature_type` attributes that enable this classification.

### Integration with Existing Transform Infrastructure

The library's `actions.py` module currently provides `rotate()`, `translate()`, and `transform()` functions that apply transformations via the sandwich product. These functions would be enhanced by exponential-based constructors:

Current pattern:
```python
# In actions.py
M = rotor(B, angle)  # Would need exponential-based implementation
result = M * b * ~M
```

With exponentials, the rotor is generated as $\mathbf{R} = e^{-\theta \mathbf{B}/2}$ where $\theta$ is the rotation angle and $\mathbf{B}$ is the normalized bivector defining the rotation plane. The result is then applied via the sandwich product $\mathbf{R} \mathbf{b} \tilde{\mathbf{R}}$.

The half-angle convention arises naturally from the bivector being the infinitesimal generator of the rotation.

### Connection to Motors and General Versors

While bivectors generate rotations, the exponential generalizes to higher grades:

- **Trivectors** in 4D and higher generate certain types of transformations
- **Even-grade blades** produce even-grade versors (generalized rotors)
- **Odd-grade blades** produce odd-grade versors (reflections, etc.)

The pattern is systematic: $e^{\mathbf{B}}$ has the same grade parity as $\mathbf{B}$. This would integrate with the planned `Motor` class mentioned in the user's memory, which unifies rotations and translations as a single versor type.

---

## Part II: Enhanced Linear Algebra Integration

### The Structure-Preservation Challenge

Traditional linear algebra operates on vectors in $\mathbb{R}^n$, flattening all structure into column vectors and matrices. Geometric algebra, by contrast, maintains rich hierarchical structure: scalars, vectors, bivectors, trivectors, and their collections.

The challenge is bridging these perspectives without losing geometric information. The morphis library needs linear operators that:

1. Respect grade structure
2. Preserve geometric meaning through numerical operations
3. Interface efficiently with scipy/numpy for solving inverse problems
4. Support batch operations on collections of blades

### Current State in Morphis

The library provides substantial linear algebra infrastructure:

**Operator Class** (`elements/operator.py`):
- Represents structured linear maps $L: V \to W$ between blade spaces
- Maintains full tensor structure (no flattening)
- Supports collection dimensions for batch operations
- Provides forward application, adjoint, SVD, pseudoinverse

**Outermorphisms** (`operations/outermorphism.py`):
- Grade-preserving linear maps that respect wedge products
- Extends vector maps to arbitrary grades via exterior powers
- Implements $\underline{f}(\mathbf{a}_1 \wedge \cdots \wedge \mathbf{a}_k) = f(\mathbf{a}_1) \wedge \cdots \wedge f(\mathbf{a}_k)$

**Example Usage** (`examples/linear_operators.py`):
- Demonstrates electromagnetic current-to-field operators
- Shows SVD decomposition preserving structure
- Illustrates pseudoinverse for least-squares problems

What's **missing**:
- Blade eigenvalue problems
- Matrix representation utilities for interfacing with external libraries
- Spectral decomposition of bivectors
- Specialized solvers exploiting geometric structure
- Direct support for common electromagnetic inverse problems

### Outermorphisms: What Exists

The current `outermorphism.py` module implements the fundamental grade-preserving maps. For a linear transformation $f: V \to V$ on vectors, its extension to $k$-vectors is:

$$
\underline{f}(\mathbf{B}) = \underline{f}(v_1 \wedge \cdots \wedge v_k) = f(v_1) \wedge \cdots \wedge f(v_k)
$$

In components, this becomes:

$$
(\underline{f}(\mathbf{B}))^{i_1 \ldots i_k} = A^{i_1}_{m_1} \cdots A^{i_k}_{m_k} B^{m_1 \ldots m_k}
$$

which is exactly $k$ copies of the matrix $A$ contracting with the blade indices—a natural einsum operation.

The implementation correctly handles:
- Arbitrary grade $k$
- Preservation of the determinant property: $\underline{f}(\mathbb{I}) = \det(f) \mathbb{I}$
- Composition: $\underline{f \circ g} = \underline{f} \circ \underline{g}$

This is solid foundational work. What's needed is extension to more specialized linear algebra scenarios.

### Matrix Representations: What's Missing

Sometimes explicit matrix representations are necessary for:

1. **Eigenvalue analysis**: scipy's `eig()` and related functions
2. **Validation**: Cross-checking einsum operations against explicit matrix math
3. **Interfacing with external tools**: Passing operators to general-purpose solvers
4. **Communication**: Some collaborators think in matrices

The geometric product $\mathbf{AB}$ can be represented as matrix multiplication. Each multivector has $2^d$ components. Representing it as a column vector:

$$
\mathbf{A} = a_0 + a_1 \mathbf{e}_1 + \cdots + a_{12\ldots d} \mathbf{e}_{12\ldots d}
\quad \leftrightarrow \quad
\begin{pmatrix} a_0 \\ a_1 \\ \vdots \\ a_{12\ldots d} \end{pmatrix}
$$

Left-multiplication by $\mathbf{A}$ becomes a $2^d \times 2^d$ matrix $L_\mathbf{A}$ where:

$$
(L_\mathbf{A})_{ij} = \text{coefficient of basis blade } i \text{ in } \mathbf{A} \cdot (\text{basis blade } j)
$$

This representation isn't currently provided but would complement the existing einsum-based operations. It would be particularly useful for small dimensions ($d \leq 4$) where $2^d$ remains manageable.

### Eigenvalue Problems for Blades

A bivector $\mathbf{B}$ in $d$-dimensions defines a linear transformation via the commutator product:

$$
\mathbf{v} \mapsto [\mathbf{B}, \mathbf{v}] = \frac{1}{2}(\mathbf{Bv} - \mathbf{vB})
$$

This transformation has eigenvectors—the vectors that lie in the rotation plane (rotated) and perpendicular to it (unchanged).

More generally, a $k$-blade $\mathbf{B}$ defines a $k$-dimensional oriented subspace. The eigenvalue problem asks: what vectors span this subspace? The existing `operations/factorization.py` module likely touches on this, but systematic eigendecomposition tools are absent.

**Practical Application**: In electromagnetic analysis, magnetic field bivectors at each spatial point have principal directions. Finding these eigenvectors reveals the dominant field orientations, useful for sensor placement and inverse problem regularization.

The implementation would:
1. Convert the bivector to its skew-symmetric matrix representation
2. Use NumPy's `eig()` to find eigenvalues (purely imaginary for bivectors)
3. Extract the 2D rotation plane from the complex eigenvector pair
4. Return as orthonormal blade basis

For 3D, the Hodge dual provides a shortcut: the dual of a bivector is the rotation axis vector. For general $d$, the full eigendecomposition is necessary.

### Application to Electromagnetic Inverse Problems

The user's specific application involves estimating circuit currents from magnetic field measurements. This is a structured linear problem:

$$
\mathbf{B}(\mathbf{x}) = \sum_i I_i \mathbf{B}_i(\mathbf{x})
$$

where:
- $\mathbf{B}(\mathbf{x})$ is the measured magnetic field bivector at position $\mathbf{x}$
- $I_i$ are the unknown scalar currents
- $\mathbf{B}_i(\mathbf{x})$ are the known basis field bivectors from each circuit element

Traditional approach: Flatten to $\mathbf{y} = A\mathbf{x}$ and solve with least-squares, losing geometric structure.

**Geometric algebra approach**: Keep the bivector structure throughout using the `Operator` class.

The operator maps scalars (currents) to bivectors (fields):

$$
\mathcal{G}: \mathbb{R}^N \to \Lambda^2(\mathbb{R}^3)
$$

In the morphis framework, this would be:
```python
G = Operator(
    data=G_data,                                    # Shape: (3, 3, M, N)
    input_spec=BladeSpec(grade=0, collection=1),    # N scalar currents
    output_spec=BladeSpec(grade=2, collection=1),   # M bivector fields
    metric=euclidean(3)
)
```

The existing `Operator` class supports this pattern. The examples show SVD decomposition and pseudoinverse solutions. What's **missing** is:

1. **Structured regularization**: Constraints respecting bivector properties (e.g., field must satisfy $\nabla \wedge \mathbf{B} = 0$)
2. **Geometric preconditioning**: Using the bivector structure to improve condition numbers
3. **Sparsity patterns**: Exploiting that field measurements are typically local
4. **Uncertainty quantification**: Bayesian approaches preserving geometric structure

### Enhancements to Operator Class

The current `Operator` class provides:
- Forward application via einsum
- Adjoint operator $L^*$
- SVD decomposition: $L = U \Sigma V^*$
- Pseudoinverse for least-squares

Proposed enhancements:

**1. Geometric Regularization**

Add methods exploiting blade structure:
- Bivector field divergence constraints (Maxwell equations)
- Orientation-preserving regularization (bivector attitude smoothness)
- Grade-specific penalties (energy minimization in each grade)

**2. Iterative Solvers**

The current implementation uses direct methods (SVD, pseudoinverse). For large problems, iterative methods are essential:
- Conjugate gradient on the normal equations
- GMRES for non-square operators
- Preconditioned methods using geometric structure

These would maintain the tensor structure throughout iterations, never flattening to traditional vectors.

**3. Block Structure Exploitation**

Many electromagnetic problems have natural block structure:
- Different current loops couple weakly
- Field measurements partition by spatial regions
- Frequency-domain problems separate by harmonic

The `Operator` class could detect and exploit this structure automatically through collection dimensions.

**4. Sensitivity Analysis**

For inverse problems, understanding how measurement errors propagate is crucial. The operator framework should provide:
- Condition number analysis per grade
- Singular vector interpretation (which bivector modes are observable)
- Optimal sensor placement (maximizing smallest singular value)

### Interface with NumPy/SciPy

The morphis library's strength is maintaining geometric structure. But sometimes interfacing with traditional linear algebra libraries is necessary. The gap to fill:

**Conversion utilities**:
```python
# Blade → column vector → Blade
def blade_to_vector(b: Blade) -> np.ndarray:
    """Flatten blade components to vector."""
    ...

def vector_to_blade(v: np.ndarray, spec: BladeSpec, metric: Metric) -> Blade:
    """Reconstruct blade from flattened components."""
    ...
```

**Operator → sparse matrix**:
```python
def operator_to_sparse(L: Operator) -> scipy.sparse.csr_matrix:
    """Convert operator to sparse matrix for scipy solvers."""
    ...
```

This would enable using scipy's extensive solver library while preserving the ability to interpret results geometrically.

### Comparison to Existing Linear Operators Module

The `examples/linear_operators.py` demonstrates the current capabilities well:

- Current-to-field transfer operators ($G^{WX}_{Kn}$ mapping scalars to bivectors)
- SVD analysis showing condition numbers and reconstruction
- Pseudoinverse solutions for least-squares

What it shows is **missing**:
- Eigenvalue analysis of the bivector fields produced
- Regularization exploiting physical constraints
- Iterative solution methods for large-scale problems
- Uncertainty quantification in the reconstructed currents

The example correctly maintains structure through the solve, never flattening blades to generic vectors. This is the right pattern. The enhancement is adding specialized tools for common problem classes:

1. **Electromagnetic transfer operators**: Standard interface for current → field, charge → potential
2. **Poisson solvers**: Field → source reconstruction with physical constraints
3. **Time-dependent problems**: Structured operators for evolution equations

### Spectral Methods for Bivectors

Bivectors in 3D correspond to skew-symmetric $3 \times 3$ matrices. Their eigenvalues are always $\{0, \pm i\omega\}$ for some real $\omega$. The eigenvectors reveal:

- The rotation plane (complex eigenvector pair)
- The rotation axis (null eigenvector)
- The rotation rate ($\omega$)

In higher dimensions, the pattern generalizes. A bivector in $d$-dimensions decomposes into orthogonal 2-planes, each with its own rotation rate.

**Implementation sketch**:

1. Convert bivector to skew-symmetric matrix representation
2. Compute eigendecomposition (purely imaginary eigenvalues)
3. Pair eigenvalues: $\pm i\omega_k$
4. Extract rotation planes from eigenvector pairs
5. Return as collection of orthonormal bivector basis elements

This would enable:
- Normal form decomposition of general bivectors
- Finding principal rotation planes in high-dimensional problems
- Simplifying bivector expressions by basis change

The existing `operations/factorization.py` module (not yet reviewed in detail) may contain related blade factorization tools, but systematic spectral decomposition is likely absent.

---

## Integration Strategy

### Module Organization

Proposed additions to the library structure:

```
operations/
    exponential.py       # exp, log, slerp for blades and multivectors
    spectral.py          # Eigenvalue problems for blades
    matrix_rep.py        # GA ↔ matrix conversions

algebra/
    decompositions.py    # SVD, QR, Schur for GA operators
    regularization.py    # Geometric regularization strategies
```

### Dependency Relationships

**Exponentials** depend on:
- `operations/products.py`: Geometric product for computing $\mathbf{B}^2$
- `operations/norms.py`: Magnitude computation
- `elements/metric.py`: Signature determination

**Spectral tools** depend on:
- `operations/exponential.py`: Log extracts bivectors from rotors
- `operations/products.py`: Commutator product for bivector action
- NumPy's `linalg.eig`: External eigenvalue solver

**Matrix representations** depend on:
- `operations/structure.py`: Basis blade enumeration
- `operations/products.py`: Geometric product evaluation

### Implementation Priority

**Phase 1: Exponentials** (highest impact)
1. Implement `exp_blade()` for grades 0, 1, 2
2. Implement `log_versor()` for even multivectors
3. Add `slerp()` interpolation function
4. Integrate with `actions.py` transformation constructors

**Phase 2: Matrix Interfaces** (enables external tools)
1. Blade ↔ vector conversion utilities
2. Geometric product as matrix operator
3. Operator → sparse matrix for scipy interface

**Phase 3: Spectral Tools** (specialized analysis)
1. Bivector eigendecomposition
2. General blade spectral analysis
3. Normal form computations

**Phase 4: Advanced Linear Algebra** (problem-specific)
1. Electromagnetic transfer operator templates
2. Regularization strategies
3. Iterative solver wrappers
4. Uncertainty quantification

---

## Expected Impact

### For Theoretical Geometric Algebra

**Exponentials** complete the Lie theory foundation:
- Explicit exp/log maps between algebra and group
- Geodesic flows on rotation manifolds
- Enables the planned `manifold/` module development

**Matrix representations** bridge to classical results:
- Validate GA operations against textbook linear algebra
- Translate classical theorems to GA language
- Enable mixed GA/matrix workflows

### For Electromagnetic Applications

**Structure-preserving operators** improve inverse problems:
- Physical constraints (Maxwell equations) built into solver
- Better conditioning through geometric preconditioning
- Interpretable results maintaining bivector field structure

**Spectral tools** enable field analysis:
- Identify dominant field orientations
- Optimize sensor placement via singular value analysis
- Reduce problem dimensionality through principal component extraction

### For Visualization and Animation

**Slerp interpolation** enables smooth animations:
- Constant angular velocity between orientations
- No gimbal lock or quaternion/Euler angle artifacts
- Natural integration with existing `animate_3d.py` and `animate_4d.py`

**Exponential parameterization** simplifies camera controls:
- Direct manipulation of rotation bivectors
- Smooth transitions between views
- Keyframe interpolation via bivector addition

---

## Conclusion

The morphis library provides a mathematically rigorous foundation for geometric algebra computation, with particular strengths in:
- Systematic einsum-based products preserving tensor structure
- Comprehensive blade and multivector operations
- Structured linear operators for batch processing
- PGA transformations and geometric constructors

The two major enhancements proposed—**exponentials/logarithms** and **enhanced linear algebra integration**—would elevate the library from a solid computational framework to a complete geometric algebra system. These additions are not peripheral features but fundamental operations that:

1. **Connect algebra to geometry**: Exponentials bridge infinitesimal (bivectors) and finite (rotors) transformations
2. **Enable numerical methods**: Structure-preserving linear operators solve real-world inverse problems
3. **Support applications**: From smooth animations to electromagnetic field analysis
4. **Maintain mathematical elegance**: All enhancements respect the library's core principle of preserving geometric structure

The implementation strategy builds incrementally on existing infrastructure, requiring no major architectural changes while substantially expanding capabilities. The result would be a geometric algebra library that is both theoretically complete and practically powerful for scientific computing applications.
