# Manifolds: Mathematical Properties

Worked submanifolds whose geometry is known in closed form, checked from the
patch alone. Derivatives are fourth-order differences, so agreement is to
about $10^{-9}$.

## Domains

- a complex region $I + i J$ contributes the two real coordinates $(\mathrm{Re} \, w, \mathrm{Im} \, w)$, adjacent
- periodic coordinates leave out their endpoint, and their cell complex closes: a cylinder has $χ = 0$ and two boundary circles

## Clifford Torus

The patch $\mathbf{X}(θ, φ) = (\cos θ, \sin θ, \cos φ, \sin φ)$ in $\mathbb{R}^4$:

```math
\begin{aligned}
h_{ab} &= δ_{ab} \\ \\
\mathbf{k}_{11} &= -\mathbf{r}_1 \\ \\
K &= 0
\end{aligned}
```

where $\mathbf{r}_1$ is the radius vector of the first circle. In $\mathbb{C}^2$,
as $(e^{iθ}, e^{iφ}) / \sqrt{2}$, every tangent plane is Lagrangian: $θ_K = π / 2$.

## Sphere

For radius $ρ$:

- $K = 1 / ρ^2$ and $R = 2 K$
- $R_{ab} = h_{ab} / ρ^2$
- $\mathbf{k} = -\mathbf{X} / ρ^2$, pointing inward
- the normal is radial and $\sqrt{\det h} = ρ^2 \sin θ$

## Projections

- $P(\mathbf{v}) + P_\perp(\mathbf{v}) = \mathbf{v}$, $P^2 = P$, and $P_\perp(\mathbf{v}) \cdot \partial_a \mathbf{X} = 0$
- the normal frame is orthonormal and orthogonal to the tangent frame
- $R_{abcd}$ is antisymmetric in each index pair

## Causal Character

In $\mathbb{R}^{1,2}$, in both conventions:

- de Sitter space, $(\sinh t, \cosh t \cos φ, \cosh t \sin φ)$, is timelike everywhere
- a slice of constant time is spacelike
- the light cone $(r, r \cos φ, r \sin φ)$ is null

## Holomorphic Curves

The Fermat curve $w^5 + z^5 = 1$ in $\mathbb{C}^2$, parametrized by $w$:

- $\partial_2 \mathbf{X} = J(\partial_1 \mathbf{X})$, so $θ_K = 0$
- $\mathbf{k} = 0$: complex curves are minimal
- $K < 0$
