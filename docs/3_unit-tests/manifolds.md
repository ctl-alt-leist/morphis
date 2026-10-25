# Manifolds: Mathematical Properties

Worked submanifolds whose geometry is known in closed form, checked from the patch alone. Derivatives are fourth-order differences, so agreement is to about $10^{-9}$.

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

where $\mathbf{r}_1$ is the radius vector of the first circle. In $\mathbb{C}^2$, as $(e^{iθ}, e^{iφ}) / \sqrt{2}$, every tangent plane is Lagrangian: $θ_K = π / 2$.

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

## Atlases

The Fermat curve $z_1^n + z_2^n = 1$ is covered by $n^2$ patches, $z_1 = e^{2πi k_1 / n} \cosh(w)^{2/n}$ and $z_2 = e^{2πi k_2 / n} (-i \sinh w)^{2/n}$. Welded, they give genus $g = (n - 1)(n - 2) / 2$ with the $n$ points at infinity cut off:

- $χ = 2 - 2 g - n$ with $n$ boundary circles, for $n = 2, 3, 5$
- every welded vertex satisfies the curve equation

## Projections

- an orthographic picture coordinate is $\mathbf{e}_c \cdot \mathbf{x}$
- stereographic projection fixes the equator, sends the antipode of the pole to the origin, and takes the Clifford torus to a torus of revolution
- composition with `@` agrees with applying the maps in sequence
- the pushforward agrees with a central difference, and `pinching` is 1 on a faithfully drawn plane and 0 on one containing the hidden direction
- `blade` under an orthographic view is the sub-array of components at the axes, and under perspective it is the wedge of the pushforwards
- every view draws a space of dimension 3 or less in its own coordinates; axes are three distinct labels, routed through the metric
- perspective from $\mathbb{R}^4$ draws $D \, \mathbf{x}_{123} / (D - x_4)$, from $\mathbb{R}^5$ divides by $1 - (x_2 + x_4) / D$ for axes $(1, 3, 5)$, and keeps collinear points collinear
- the stereographic view agrees with `Projection.stereographic`, reorders and drops directions by label, and refuses a pole among its axes
- in the Scene, a frame's corners are the images of the true corners (not sums of the images of its vectors), an arrow's tip is the image of its tip, and a tesseract's cell has 24 faces
