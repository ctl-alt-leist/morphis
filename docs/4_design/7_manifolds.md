# Manifolds, Topology, and Projection

This document is the design for curved objects in morphis: submanifolds of a
flat space of any dimension and signature, the topology they carry, the complex
structures they may respect, and the maps that draw them in three dimensions.

The aim is to make the mathematics of the physics notebook executable in the
form it is written. The references are *Higher Dimensional Manifolds* in the
geometric algebra chapter, for submanifolds and projection; the Kähler chapter
and appendix of *Where Quantum States Live*, for complex structure; and *Real
Gauge Geometry*, for bundles and transport. The theoretical backbone is the
vector-manifold calculus of Hestenes and Sobczyk (*Clifford Algebra to
Geometric Calculus*, chapters 4 and 5), in which a manifold is described
entirely by objects of the ambient algebra.

## Principles

**A manifold is a map, and the code reads $\mathbf{X}(σ)$.** A patch is a
callable on coordinates. Every geometric quantity is a function of the point,
named after the mathematical object it computes: `X(σ)`,
`X.tangent_frame(σ)`, `X.induced_metric(σ)`, `X.second_fundamental_form(σ)`.
Sampling a manifold is calling it on a grid of coordinates.

**The lot is the coordinate grid.** A grid of $σ$ is a lot of shape $(N_1,
\ldots, N_k)$, one axis per intrinsic coordinate, and lot axis $a$ is the $σ^a$
direction. Every operation in morphis already broadcasts over it. A field on a
manifold is therefore a `Vector` whose lot is the grid; no separate field class
is needed until bundles need sections.

**Everything lives in the ambient algebra.** The tangent frame $\partial_a
\mathbf{X}$, the tangent blade $\mathbf{t}$, the projector $P$, the normal space,
and the second fundamental form $\mathbf{k}_{ab}$ are all ambient objects, so
one formula holds in every dimension and signature. Quantities with manifold
indices, such as $h_{ab}$, are plain $k \times k$ arrays per point. They are not
`Metric` objects, because a `Metric` is a constant flat context.

**Frames compute; blades are views.** morphis stores a grade-$r$ element with
all $d^r$ components:

| Object | Components per point |
|--------|----------------------|
| tangent 2-blade in $\mathbb{R}^4$ | 16 |
| tangent 3-blade in $\mathbb{R}^6$ | 216 |
| tangent 6-blade in $\mathbb{R}^8$ (quintic threefold) | 262 144 |

The tangent `Frame`, of shape $k \times d$, is what the code computes with.
$\det h$ is its Gram determinant, $P(\mathbf{v}) = h^{ab} \, (\mathbf{v} \cdot
\partial_b \mathbf{X}) \, \partial_a \mathbf{X}$, the normal space is its metric
null space, and the Kähler angle is read from it directly. The blade
$\mathbf{t}$ is built when asked for.

**The algebra specifies; the closed form implements.** Where a geometric
algebra formula is exponential in $d$ when evaluated densely, the code uses the
equivalent closed form, and a test checks the two agree in low dimension. The
perspective meet $(\mathbf{c} \wedge \mathbf{p}) \vee \mathbf{s}$ is the main
case.

**Topology is separate from geometry.** Cells are integer arrays that know
nothing of dimension, metric, or position. One cell complex serves a torus in
$\mathbb{R}^3$, in $\mathbb{R}^4$, and in $\mathbb{R}^{2,2}$.

**The metric decides geometry; the projection decides the picture.** The metric
determines $h_{ab}$, the normal space, curvature, and causal character. The
picture comes from an explicit projective map to drawing coordinates. The
metric may help construct the map, as for an orthonormal screen, but is never
consulted when the map is applied.

**The space carries its structure.** Signature, block order, index labels, and
the pairing of directions into modes belong to the space, which in morphis is
the `Metric`. The pseudoscalar, the structure bivector, and the labels shown in
display are derived from it.

## The Space

A space is declared by its blocks, in a fixed order:

```python
metric(space=3)                          # Euclidean ℝ³
metric(space=3, projective=1)            # PGA
metric(time=1, space=3)                  # spacetime
metric(time=1, space=3, projective=1)    # spacetime PGA
metric(modes=2)                          # ℂ² as ℝ⁴
```

Storage follows the blocks: (time, space, projective) in general, and
(projective, space) for a purely spatial space with a single projective
direction. Directions are labeled by the number their position gives them, with
the first spatial direction always 1:

| Space | Order | Labels |
|-------|-------|--------|
| Euclidean | space | $1, \ldots, n$ |
| PGA | projective, space | $0, 1, \ldots, n$ |
| Spacetime | time, space | $0, 1, \ldots, n$ |
| Spacetime PGA | time, space, projective | $0, 1, \ldots, n, n + 1$ |
| Kähler | modes | $1, \dot 1, 2, \dot 2, \ldots$ |

The sign convention is an argument of its own, since the geometric algebra
chapter uses $(+, -, -, -)$ and the gauge geometry book uses $(-, +, +, +)$.
The `Metric` reports its signature as the inertia $(p, q, r)$ of the tensor, and
two metrics are equal when their layouts and tensors agree, so $\mathbb{R}^{2,2}$
and $\mathbb{R}^{1,3}$ differ, and so do $\mathbb{R}^4$ and $\mathbb{C}^2$. In
code the partner label $\dot m$ is `Dot(m)`.

The block layout leaves room for later cases without designing them now: a
second time direction (labeled $-1$), conformal blocks, and several projective
directions.

## Complex Structure

Two complex objects are native, and they are kept distinct.

**Complex coefficients**, $\mathbb{C} \otimes \mathrm{Cl}(V)$: the scalar $i$
is external and commutes with everything. morphis already admits complex
components.

**Complex dimensions**, $\mathbb{R}^{2n}$ with a structure bivector: the unit
$i$ is the bivector $\mathbf{J}$, an element of the algebra. A space declared
with modes pairs each representative direction $\mathbf{e}_m$ with its partner
$\mathbf{e}_{\dot m}$, in the adjacent order $1, \dot 1, 2, \dot 2, \ldots$, and
carries

$$
\begin{aligned}
\mathbf{J} &= \mathbf{e}_m \wedge \mathbf{e}_{\dot m} \\ \\
J(\mathbf{v}) &= \mathbf{v} \lrcorner \mathbf{J}
\end{aligned}
$$

Every real vector of such a space has a complex reading: `v.amplitudes` returns
the $n$ mode amplitudes $v^m + i \, v^{\dot m}$, just as `.on[...]` returns real
components, and `Vector.from_amplitudes(z, metric)` builds the real vector from
them. The amplitude carries no normalization factor by default; the convention
that places $1 / \sqrt{2}$ in the basis is an option. Holomorphic maps are
written in ordinary complex arithmetic and land as real vectors.

The vocabulary follows the Kähler appendix:

| Mathematics | Code |
|-------------|------|
| $\mathbf{J}$ | `structure_bivector(metric)` |
| $J$ | `complex_structure(metric)`, an `Operator` |
| $\cos θ = \mathbf{b} \cdot \mathbf{J}$ | `kahler_angle(b, J)` |
| $J \mathbf{b} = \mathbf{b}$ | `is_complex(b, J)` |
| $\mathbf{J} \lrcorner \mathbf{b} = 0$ | `is_isotropic(b, J)` |
| isotropic of grade $n$ | `is_lagrangian(b, J)` |

`kahler_angle` accepts a 2-blade or a tangent frame, so it applies to a surface
without building its blade.

## Topology

`CellComplex` holds $r$-cells as integer arrays.

- `grid(shape, periodic)` builds the complex of a coordinate grid, any axis
  closed into a circle. Quadrilateral cells are kept; triangulation happens only
  when a renderer asks.
- `quotient(identifications)` glues cells, and `quotient_coincident(positions,
  tolerance)` finds the identifications geometrically, which is how many patches
  become one surface.
- `boundary()`, `skeleton(r)`, `euler_characteristic()`.

## Manifolds

`Domain` describes the coordinate region: `Interval`, `Circle`, their products,
and complex regions such as `Interval(0, π/2) + 1j * Interval(-1, 1)`. A complex
coordinate $w = σ^1 + i \, σ^2$ is native, so a holomorphic patch is written in
$w$ and its real coordinates are derived.

`Patch(map, domain)` is a parametrization $\mathbf{X}: U \to V$. Derivatives
come from the map when it supplies them and from fourth-order central
differences otherwise, respecting periodic coordinates. On a holomorphic patch,
$\partial_2 \mathbf{X} = J(\partial_1 \mathbf{X})$ holds by construction.

| Mathematics | Code |
|-------------|------|
| $\mathbf{X}(σ)$ | `X(σ)` |
| $\partial_a \mathbf{X}$ | `X.tangent_frame(σ)` |
| $\mathbf{t}$ | `X.tangent_blade(σ)` |
| $h_{ab}$ | `X.induced_metric(σ)` |
| $\sqrt{\lvert \det h \rvert}$ | `X.volume_element(σ)` |
| $P$ | `X.tangent_projector(σ)` |
| normal space | `X.normal_frame(σ)`, `X.normal_blade(σ)` |
| $\mathbf{k}_{ab}$ | `X.second_fundamental_form(σ)` |
| $\mathbf{k}$ | `X.mean_curvature(σ)` |
| $K$ | `X.gaussian_curvature(σ)` |
| causal character | `X.causal_character(σ)` |

The intrinsic dimension is `span`, as for `Frame`; `dim` is always the ambient
dimension. Causal character is read from the inertia of $h_{ab}$ (its counts of
positive, negative, and zero eigenvalues), which is correct in either sign
convention.

`X.grid(shape)` returns a coordinate grid over the domain, and `X.mesh(shape)`
returns the sampled positions with their `CellComplex`.

`Atlas` is a set of patches with the complex that glues them, so a many-sheeted
or closed manifold is one object.

`LevelSet(f)` is a manifold given as $f^p(\mathbf{x}) = 0$. Its normal frame
comes from the gradients $\nabla f^p = \mathbf{e}^m \, \partial_m f^p$, and a
Newton step $\mathbf{x}' = \mathbf{x} - D^{+} f(\mathbf{x})$ moves points onto
it. The geometric quantities share the patch's names, evaluated at ambient
points instead of coordinates.

`transport(frame, curve, X)` carries a frame along a curve by the Levi-Civita
rule of an embedded manifold: step forward, project onto the next tangent
space, re-orthonormalize. The holonomy of a closed curve is the rotor relating
the final frame to the initial one.

## Mesh

`Surface` becomes `Mesh`: vertices as a grade-1 `Vector` of any dimension and
metric, cells from a `CellComplex`. The file loaders remain as constructors. A
mesh is what passes from geometry to drawing; it carries no derivatives.

## Projection

A projective map from $\mathbb{RP}^d$ to $\mathbb{RP}^3$ is a $4 \times (d + 1)$
matrix whose kernel is the center of projection, a $(d - 4)$-flat. This is the
closed form of $(\mathbf{c} \wedge \mathbf{p}) \vee \mathbf{s}$. Orthographic
projection has an ideal center, and a linear map has a constant weight row.
`Projection` holds the matrix and replaces both `ProjectionConfig` and the ink
module's `Depiction`:

- `Projection.orthographic(screen)` from a 3-blade or an orthonormal `Frame`.
- `Projection.perspective(center, screen)` from flats.
- `Projection.stereographic(radius, pole)`: perspective from a point of the
  sphere onto the hyperplane through its center perpendicular to the pole.
- `Projection.linear(images)`: the deliberately unfaithful depiction, each
  direction sent to a chosen drawing vector.
- Composition with `@`, so a stereographic map from $S^{d-1}$ followed by an
  orthographic one reaches 3D from any $d$.

Operations:

- `Π(v)` maps vectors of any lot, dividing by the weight once, at the end.
- `Π.turned(R)` turns the screen by a rotor, equivalent to turning the object
  by $\tilde{R}$.
- `Π.hidden(v)` returns the coordinates the picture discards.
- `Π.blade(b)` maps a blade by the outermorphism.
- `Π.pinch_points(X, σ)` finds where the differential of the map drops rank on
  the tangent frame; for an orthographic map this is
  $\bar{P}_{\mathbf{s}}(\mathbf{t}) = 0$.

Both renderers, the PyVista `Scene` and the ink `Sketch`, take meshes and curves
in drawing coordinates with an optional lot-shaped `Vector` for color.

For $k \ge 3$, the coordinate 2-skeletons are drawn, or the manifold is sliced
by a moving hyperplane and the slices drawn in sequence.

## Bundles

Deferred until the layers above are done, and shaped by them: a bundle over a
sampled base is an `Atlas` of local products, a section is a lot-shaped
`Vector`, a transition function is a field of rotors on the overlap, and
parallel transport is frame transport with a connection bivector in place of
the projector.

## Acceptance

Each layer is tested against worked examples from the notebook, and each
complete figure is also an example script.

| Test | Expected |
|------|----------|
| periodic grid | torus, $χ = 0$ |
| Clifford torus | $h_{ab} = δ_{ab}$, $K = 0$, $\mathbf{k}_{11} = -\mathbf{r}_1$ |
| sphere of radius $ρ$ | $K = 1 / ρ^2$ |
| de Sitter in $\mathbb{R}^{1,2}$ | timelike everywhere, in both conventions |
| Fermat curve, $n = 5$ | $\cos θ_K = 1$, $\mathbf{k} = 0$, welded mesh $χ = -15$ with 5 boundary circles |
| Fermat projection at $α = π/4$ | exactly 15 pinch points, at the derived locations |
| perspective | agrees with the PGA meet in $d = 4, 5$ |
| stereographic Clifford torus | a torus of revolution |
| qubit phase orbit | the $(1, 1)$ curve on the torus of constant populations |
| transport around colatitude $θ_0$ | rotation by $2π \, (1 - \cos θ_0)$ |

## Phases

1. The space: block declaration, explicit signature and convention, labels,
   modes, equality by tensor.
2. Complex structure: amplitudes, structure bivector, blade classification,
   Kähler angle.
3. `CellComplex`.
4. `Domain`, `Patch`, and the submanifold quantities.
5. `Projection` and `Mesh`; renderers consume both.
6. Examples: Clifford torus, Fermat curve, qubit phase orbit, de Sitter.
7. `Atlas`, `LevelSet`, transport, and the first bundle figures.
8. Concepts documentation in `docs/1_concepts/`.
