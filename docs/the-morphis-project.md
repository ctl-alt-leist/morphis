# The Morphis Project

## Vision

**Morphis** is a unified mathematical framework for geometric computation, providing elegant tools for working with geometric algebra, manifolds, and their applications across mathematics and physics.

The name derives from Greek μορφή (morphe, "form") — embodying the transformation and adaptation of geometric structures across different contexts while preserving their essential nature. Like its mythological parents Proteus (shape-shifting transformation) and Atlas (structure-bearing), morphis provides both the flexibility to work across diverse geometric contexts and the rigorous mathematical structure to maintain correctness.

### Core Philosophy

**Geometric algebra as computational backbone.** Every geometric structure — from simple vectors to complex manifolds — is expressed through the language of geometric algebra, providing unified operations (wedge products, geometric products, duality) that work consistently across contexts.

**Context-aware but flexible.** Geometric objects know their context (Euclidean, projective, conformal, spacetime) when it matters, but core algebraic operations work regardless of context. This enables both safety (context validation where needed) and flexibility (mix contexts when appropriate).

**Mathematical structures, not applications.** The package provides the geometric and algebraic tools; applications (specific physical systems, engineering problems) live in examples and downstream packages.

---

## Package Structure

```
morphis/
├── algebra/           # Abstract algebraic structures
├── ga/                # Geometric algebra (the computational core)
├── manifold/          # Differential geometry on curved spaces
├── geometry/          # Geometric contexts (PGA, CGA, spacetime, etc.)
├── topology/          # Topological structures and invariants
├── visualization/     # Geometric visualization tools
└── utils/             # Shared utilities
```

### Conceptual Hierarchy

The packages form a natural dependency hierarchy from abstract to concrete:

```
Foundation
└── algebra           # Groups, representations, abstract algebras

Core Framework  
└── ga                # Geometric algebra: blades, operations, transforms

Mathematical Structures
├── manifold          # Charts, connections, curvature
└── topology          # Homology, homotopy, bundles

Geometric Contexts
└── geometry          # Euclidean, projective, conformal, spacetime, symplectic

Infrastructure
├── utils             # Shared utilities and helpers
└── visualization     # Plotting and visual representation
```

---

## Key Architectural Decisions

### 1. Geometric Algebra as Foundation

**Decision:** All geometric computations use GA structures (blades, multivectors) as the primary representation.

**Rationale:** 
- Unified framework: wedge products, geometric products, duality work everywhere
- Coordinate-free: geometric meaning independent of basis choice
- Efficient: einsum-based operations on antisymmetric tensors
- Extensible: naturally handles different signatures and dimensions

**Implication:** Even curved manifolds use GA in local tangent spaces; transformations expressed as GA operations (rotors, motors, versors).

---

### 2. Context Management

**Decision:** Geometric context is intrinsic to blade objects, combining signature and structure.

**Structure:** Two orthogonal axes define context:

**Signature Axis** (metric structure):
- Euclidean: positive definite metric (+++)
- Lorentzian: spacetime signature (+---)
- Degenerate: null direction for PGA (0+++)

**Structure Axis** (geometric interpretation):
- Flat: standard geometric algebra
- Projective: ideal points, incidence relations
- Conformal: angles, circles, inversions  
- Round: projective + conformal combined

**Examples:**
- `euclidean.flat` — Standard Euclidean GA
- `euclidean.conformal` — Conformal GA (CGA)
- `degenerate.projective` — Flat projective GA (PGA)
- `degenerate.round` — Round projective GA
- `lorentzian.flat` — Spacetime GA

**Rationale:**
- Systematic: all contexts follow consistent naming
- Compositional: signature and structure are independent concerns
- Extensible: easy to add new signatures or structures
- Type-safe: validation where needed, flexible where appropriate

---

### 3. Two Types of Transformations

**Decision:** Distinguish geometric transformations (GA operations) from coordinate transformations (manifold operations).

**Geometric Transformations** (`morphis.ga.transforms`):
- Rotations via rotors: $R = e^{-\theta B/2}$, apply as $RxR^{-1}$
- Translations via motors (PGA): $T = 1 + \frac{1}{2}e_0 \wedge d$
- Reflections: $-nxn^{-1}$
- Inversions (CGA): conformal transformations

**Coordinate Transformations** (`morphis.manifold.transforms`):
- Spherical ↔ Cartesian conversions
- Null coordinates for spacetime
- Penrose compactification
- General diffeomorphisms between charts

**Rationale:**
- Clear separation of concerns
- Geometric transformations preserve structure using GA
- Coordinate transformations just relabel points on manifolds
- Different mathematical objects, different modules

---

### 4. Mathematics vs Applications

**Decision:** Package contains mathematical structures; specific applications live in examples.

**In Package:**
- Lorentzian metric and causal structure → `morphis.geometry.spacetime`
- Symplectic forms and Poisson brackets → `morphis.geometry.symplectic`
- Curvature tensors and connections → `morphis.manifold.connection`
- Generic geometric objects (circles, spheres, planes) → `morphis.geometry.*`

**In Examples:**
- Schwarzschild black hole solutions
- Harmonic oscillator Hamiltonian
- Electromagnetic field configurations
- Specific mechanical systems

**Rationale:**
- Package provides tools, not solutions
- Mathematical structures are reusable; applications are specific
- Keeps package focused and maintainable
- Users build applications using morphis tools

---

### 5. No Premature Abstraction

**Decision:** Create modules when duplication emerges, not preemptively.

**Examples:**
- `tensor.py`: Only if general (p,q) tensors needed beyond blades
- `linear_algebra.py`: Only if matrix operations used across multiple modules
- Specialized operations: Start in geometry modules, extract if shared

**Rationale:**
- Avoid speculative generality
- Let actual usage patterns drive structure
- Easier to combine than to split later
- Keep initial codebase lean and focused

---

## Package Details

### morphis.algebra

Abstract algebraic structures that underlie geometric computations.

**Contents:**
- Groups: discrete groups, Lie groups, group actions
- Representations: linear representations, adjoint representations
- Abstract algebras: general algebraic operations and structures

**Not included:** Clifford algebras (that's just GA with different terminology)

**Purpose:** Provide algebraic foundations for symmetries, transformations, and abstract structure.

---

### morphis.ga

The computational heart of morphis — geometric algebra operations and structures.

**Contents:**
- `model.py`: Blade, MultiVector, Metric data structures
- `context.py`: Signature, Structure, GeometricContext
- `operations.py`: wedge product, interior product, projection
- `duality.py`: complements, Hodge dual
- `norms.py`: norm, normalize operations
- `structure.py`: Levi-Civita symbols, einsum signature generation
- `utils.py`: antisymmetrization, broadcasting utilities
- `products.py`: (planned) contraction operations, geometric product
- `transforms.py`: (planned) rotors, motors, versors, reflections

**Core Principle:** Context-agnostic operations that work on any blade, with context preserved or merged appropriately.

**Key Features:**
- Full antisymmetric tensor storage (all d^k components)
- Collection dimension support (broadcasting)
- Einsum-based operations for efficiency
- Optional metric for metric-dependent operations

---

### morphis.manifold

Differential geometry on curved spaces.

**Contents:**
- `chart.py`: coordinate systems and domains
- `atlas.py`: compatible collections of charts
- `tangent.py`: tangent and cotangent bundles
- `connection.py`: covariant derivatives, parallel transport
- `curvature.py`: Riemann tensor, Ricci tensor, scalar curvature
- `transforms.py`: coordinate transformations (spherical↔cartesian, etc.)

**Purpose:** Provide tools for working with curved spaces, enabling GA to operate in local tangent spaces while respecting global manifold structure.

**Integration with GA:** Tangent vectors are blades; forms are multivectors; parallel transport uses GA operations in local frames.

---

### morphis.geometry

Context-specific geometric operations and constructions.

**Organization:** One module per primary geometric context, with operations supporting both basic and extended structures.

**Contents:**
- `euclidean.py`: standard Euclidean operations
- `projective.py`: points, lines, planes, incidence (flat and round PGA)
- `conformal.py`: circles, spheres, inversions (CGA)
- `spacetime.py`: causal structure, light cones, Lorentzian operations
- `symplectic.py`: symplectic forms, Poisson brackets, canonical transformations

**Design Pattern:** Each module provides:
- Constructors that create blades with appropriate context
- Operations specific to that geometric interpretation
- Validation when context matters for correctness

**Example:**
```python
from morphis.geometry import projective as pga

# Creates blade with degenerate.projective context
p = pga.point([1, 0, 0])
q = pga.point([0, 1, 0])

# PGA-specific operation
L = pga.line(p, q)

# Works with round PGA too
sphere = pga.sphere([0, 0, 0], radius=1.0, round=True)
```

---

### morphis.topology

Topological structures and invariants.

**Contents:**
- `homology.py`: chain complexes, boundary operators, homology groups
- `homotopy.py`: fundamental group, higher homotopy groups
- `bundles.py`: fiber bundles, sections, connections

**Status:** Lower priority initially; implement as needed for specific applications.

---

### morphis.visualization

Geometric visualization tools.

**Contents:** Specialized visualization for different geometric objects and contexts, going beyond simple plotting:
- Blade visualization (vectors, bivectors, trivectors)
- Manifold visualization (charts, coordinate grids)
- Penrose diagrams for spacetime
- Interactive 3D visualization
- Dimension reduction for high-dimensional objects

**Not just `plotting.py`:** Different visualization strategies for different geometric objects and contexts.

---

### morphis.utils

Shared utilities that don't belong in other packages.

**Contents:** Common helper functions, data structures, and tools used across multiple modules.

**Philosophy:** Utilities emerge from actual usage; don't populate preemptively.

---

## Migration Strategy

### Current State

Existing code in `maxwell.maths`:
- `ga_model.py`: Blade, MultiVector, Metric
- `ga_operations.py`: wedge, interior, norms, projections
- `ga_projective.py`: PGA-specific operations
- `ga_utils.py`: Levi-Civita, antisymmetrization

### Migration Path

**Phase 1: Foundation**
1. Create `morphis` package structure
2. Rename `ga_model.py` → `morphis.ga.model`
3. Migrate `ga_utils.py` → `morphis.ga.utils`
4. Add context system in `morphis.ga.context`

**Phase 2: Core Operations**
1. Migrate `ga_operations.py` → `morphis.ga.operations`, `morphis.ga.duality`
2. Split out products into `morphis.ga.products`
3. Add geometric product implementation
4. Create `morphis.ga.transforms` with rotors, motors

**Phase 3: Geometric Contexts**
1. Migrate `ga_projective.py` → `morphis.geometry.projective`
2. Add context markers to all constructors
3. Implement conformal operations → `morphis.geometry.conformal`
4. Add Euclidean-specific operations → `morphis.geometry.euclidean`

**Phase 4: Extensions**
1. Add `morphis.manifold` basics (charts, atlas)
2. Add `morphis.geometry.spacetime` for Lorentzian operations
3. Implement visualization basics
4. Create comprehensive examples

**Guiding Principle:** Incremental migration with working code at each step.

---

## Design Patterns

### Context-Aware Constructors

Geometry modules provide constructors that set appropriate context:

```python
# morphis/geometry/projective.py
from morphis.ga.context import degenerate

def point(x, cdim=0):
    """Create PGA point with appropriate context."""
    # ... construct data with e₀ = 1 ...
    return Blade(data, grade=1, dim=d+1, cdim=cdim, context=degenerate.projective)
```

### Context-Agnostic Operations

Core GA operations work regardless of context:

```python
# morphis/ga/operations.py
from morphis.ga.context import GeometricContext

def wedge(*blades):
    """Wedge product - works on any blades."""
    # ... compute wedge product ...
    # Merge contexts: matching preserved, mismatched becomes None
    merged_context = GeometricContext.merge(*[u.context for u in blades])
    return Blade(result, ..., context=merged_context)
```

### Context Validation

Geometry operations validate context when correctness depends on it:

```python
# morphis/geometry/projective.py

def meet(a, b):
    """PGA meet operation - requires projective context."""
    if a.context.signature != Signature.DEGENERATE:
        raise ValueError(f"Expected projective, got {a.context}")
    # ... compute meet via duality ...
```

### Metric Passing

Metric-dependent operations take optional metric parameter:

```python
from morphis.ga import norm, interior

# Uses Euclidean metric by default
n = norm(blade)

# Can pass custom metric
n = norm(blade, metric=pga_metric)

# Interior product always needs metric
result = interior(u, v, metric=g)
```

---

## Example Workflows

### Projective Geometry

```python
from morphis.geometry import projective as pga

# Create points
p = pga.point([0, 0, 0])
q = pga.point([1, 0, 0])
r = pga.point([0, 1, 0])

# Construct geometric objects
line = pga.line(p, q)
plane = pga.plane(p, q, r)

# Geometric queries
dist = pga.distance(r, line)
incident = pga.point_on_line(r, line)

# Transformations
translated = pga.translate(p, direction=[1, 1, 0])
```

### Conformal Geometry

```python
from morphis.geometry import conformal as cga

# Create geometric objects
p = cga.point([1, 0, 0])
circle = cga.circle(center=[0, 0, 0], radius=2.0)
sphere = cga.sphere(center=[1, 1, 1], radius=1.5)

# Conformal transformations
inverted = cga.inversion(p, sphere)
```

### Spacetime Physics

```python
from morphis.geometry import spacetime as st
from morphis.ga import wedge

# Events in spacetime
event1 = st.event(t=0, x=0, y=0, z=0)
event2 = st.event(t=1, x=1, y=0, z=0)

# Causal relationships
separation = st.interval(event1, event2)
timelike = st.is_timelike(separation)

# Light cones
future_cone = st.future_light_cone(event1)
```

### Mixed Context Operations

```python
from morphis.ga import wedge, geometric_product
from morphis.geometry import projective as pga

# PGA objects
p = pga.point([1, 0, 0])
q = pga.point([0, 1, 0])

# Core GA operations work across contexts
line = wedge(p, q)  # Preserves PGA context

# Can work with generic blades too
from morphis.ga import Blade
generic_blade = Blade.vector([1, 1, 0])

# Result has generic context
mixed = wedge(p, generic_blade)  # Context becomes None
```

---

## Testing Philosophy

### Unit Tests

Each module tested independently:
- `test_ga/`: test all GA operations (wedge, interior, products, duality)
- `test_geometry/`: test each geometric context
- `test_manifold/`: test charts, connections, curvature

### Property-Based Tests

Use Hypothesis to verify algebraic properties:
- Wedge product anticommutativity
- Wedge product associativity
- Geometric product properties
- Metric compatibility

### Integration Tests

Test cross-module functionality:
- GA operations on manifolds
- Context preservation through operations
- Transformation compositions

### Benchmarks

Performance tests for critical operations:
- Single blade operations (< 1μs target)
- Batch operations (linear scaling)
- Large-dimension computations

---

## Future Directions

### Near Term (Year 1)
- Complete GA core with geometric product
- Full PGA and CGA implementations
- Basic manifold support (charts, connections)
- Spacetime operations
- Comprehensive examples

### Medium Term (Year 2)
- Fiber bundles and gauge theory
- Advanced topology (cohomology, characteristic classes)
- Symplectic mechanics
- Enhanced visualization (interactive 3D, animations)

### Long Term (Year 3+)
- Spin structures and spinors
- Twistor theory
- Quantum field theory structures
- High-performance backends (GPU, Rust)
- Domain-specific applications (robotics, graphics, ML)

---

## Contributing Guidelines

### Code Style
- Follow PEP 8
- Use type hints throughout
- Pydantic models for all data structures
- Comprehensive docstrings with mathematical notation

### Mathematical Notation
- Prefer mathematical variable names: `u, v, w` for blades; `m, n, p, q` for indices
- Use Einstein summation convention
- Document operations with both code and mathematical formulas
- Include references to relevant mathematical literature

### Testing Requirements
- Unit tests for all new functions
- Property tests for algebraic operations
- Integration tests for cross-module features
- Benchmarks for performance-critical code

### Documentation
- Docstrings for all public functions
- Tutorial notebooks for new features
- Mathematical explanations in documentation
- Examples showing practical usage

---

## Resources

### Mathematical Background
- *Geometric Algebra for Computer Science* (Dorst, Fontijne, Mann)
- *Geometric Algebra for Physicists* (Doran, Lasenby)
- *Projective Geometric Algebra* (Gunn)
- *Lectures on Differential Geometry* (Spivak)

### Implementation References
- GAmphetamine (Rust GA library)
- ganja.js (JavaScript GA visualization)
- clifford (Python Clifford algebra library)

### Community
- Project repository: [To be established]
- Issue tracker: [To be established]
- Discussions: [To be established]

---

## Conclusion

**Morphis** aims to be the definitive Python library for geometric algebra and its applications — elegant, rigorous, performant, and extensible. By providing a unified framework that scales from simple Euclidean geometry to advanced manifold theory and physics applications, it enables researchers, engineers, and students to work with geometric structures in their natural language.

The project embodies the shape-shifting spirit of its namesake: transforming across contexts while preserving essential mathematical structure, just as geometric algebra provides a unified language that adapts to diverse geometric settings while maintaining computational consistency.
