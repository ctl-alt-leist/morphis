# Position and Orientation Decomposition

## Constructing Flats from Position and Orientation

In projective geometric algebra, every $k$-dimensional flat (subspace) can be specified by two pieces of information: where it is located (position) and how it is oriented (direction). This section shows how to construct the PGA blade representing such a flat and how to extract these components from an existing blade.

Consider $d$-dimensional Euclidean space embedded in $(d+1)$-dimensional projective space with homogeneous coordinates indexed by $\{0, 1, \ldots, d\}$, where index 0 corresponds to the degenerate direction. We want to construct a $k$-dimensional flat passing through a specified point with a specified orientation.

### The Construction Formula

Given:
- **Position**: A Euclidean point $\mathbf{x} = x^m \mathbf{e}_m$ where $m \in \{1, \ldots, d\}$
- **Orientation**: A Euclidean $k$-blade $\mathbf{p} = p^{m \ldots n} \mathbf{e}_{m \ldots n}$ where all indices range over $\{1, \ldots, d\}$

The PGA blade representing the $k$-flat is constructed via the wedge product:

$$\mathbf{b} = \mathbf{q} \wedge \mathbf{p}$$

where $\mathbf{q} = \mathbf{e}_0 + x^m \mathbf{e}_m$ is the embedded PGA point. This produces a $(k+1)$-blade in projective space.

Expanding the wedge product:

$$
\begin{align}
	\mathbf{b} &= (\mathbf{e}_0 + x^m \mathbf{e}_m) \wedge (p^{a \ldots b} \mathbf{e}_{a \ldots b}) \\ \\
		&= p^{a \ldots b} \mathbf{e}_{0 a \ldots b} 
			+ x^m p^{a \ldots b} (\mathbf{e}_m \wedge \mathbf{e}_{a \ldots b})
\end{align}
$$

where indices $a, b$ range over $\{1, \ldots, d\}$ and the second term involves $(k+1)$ indices total.

### Component Formulas

The components of $\mathbf{b}$ split naturally into two types:

**Mixed components** (containing index 0):

$$b^{0 a \ldots b} = p^{a \ldots b}$$

The orientation blade directly becomes the mixed component. These components encode which $k$-dimensional subspace of the Euclidean directions the flat spans.

**Euclidean components** (all indices from $\{1, \ldots, d\}$):

$$b^{m \ldots n} = x^c p^{a \ldots b} \, \delta^{m \ldots n}_{c a \ldots b}$$

where $\delta$ is the generalized Kronecker delta that antisymmetrizes over the $(k+1)$ indices. These components encode the flat's position—specifically, how the oriented object is offset from the origin, analogous to the moment of a line.

For computational purposes, this can be written using the antisymmetric symbol:

$$b^{m \ldots n} = \frac{1}{(k+1)!} x^c p^{a \ldots b} \, \varepsilon^{c a \ldots b m \ldots n}$$

where the antisymmetric symbol has $(k+1) + (k+1) = 2k+2$ total indices.

### Extracting Position and Orientation

Given a blade $\mathbf{b} = b^{m \ldots n} \mathbf{e}_{m \ldots n}$ representing a $k$-flat in PGA, we can extract the orientation and position information by identifying the component structure.

**Extracting orientation** $\mathbf{p}$:

The orientation is directly available from the mixed components:

$$p^{a \ldots b} = b^{0 a \ldots b}$$

where all indices $a, b$ range over $\{1, \ldots, d\}$. This gives the Euclidean $k$-blade representing the flat's orientation.

**Extracting position** $\mathbf{x}$:

The position requires more care. For a $k$-flat, we need to identify a point that lies on the flat. One systematic approach uses the fact that the Euclidean components encode the moment structure.

For a line ($k=1$), the position can be recovered from:

$$x^m = \frac{\varepsilon^{mnc} b^{nc} b^{0c}}{b^{0a} b^{0a}}$$

This finds the point on the line closest to the origin by computing the moment vector crossed with the direction vector and normalizing.

For higher-dimensional flats ($k \geq 2$), the position extraction becomes more involved. One approach is to find the projection of the origin onto the flat by solving the incidence equations. Given the normal space to the flat (which can be computed via the complement), project the origin onto the flat using:

$$\mathbf{x} = \text{project}_{\mathbf{b}}(\mathbf{e}_0)$$

This requires computing the interior product operations to find the closest point on the flat to the origin.

### Example: Line in 3D

Consider constructing a line in 3D projective space ($d=3$, PGA dimension 4) through point $\mathbf{x} = (x^1, x^2, x^3)$ with direction $\mathbf{v} = v^m \mathbf{e}_m$.

**Construction**:

$$
\begin{align}
	\mathbf{l} &= (\mathbf{e}_0 + x^m \mathbf{e}_m) \wedge (v^n \mathbf{e}_n) \\ \\
		&= v^n \mathbf{e}_{0n} + x^m v^n (\mathbf{e}_m \wedge \mathbf{e}_n)
\end{align}
$$

**Components**:

Mixed (direction):
$$l^{0n} = v^n$$

Euclidean (moment):
$$l^{mn} = \frac{1}{2}(x^m v^n - x^n v^m)$$

This is precisely the moment formula $\mathbf{m} = \mathbf{x} \times \mathbf{v}$.

**Extraction**:

From $\mathbf{l}$:
- Direction: $v^m = l^{0m}$
- Position: $x^m = \varepsilon^{mnp} l^{np} l^{0p} / (l^{0a} l^{0a})$

This recovers the point on the line closest to the origin.

### Example: Plane in 4D

Consider a 2-dimensional plane in 4D Euclidean space ($d=4$, PGA dimension 5) through point $\mathbf{x} = (x^1, x^2, x^3, x^4)$ with orientation bivector $\mathbf{B} = B^{mn} \mathbf{e}_{mn}$ where $m,n \in \{1,2,3,4\}$.

**Construction**:

$$\mathbf{p} = (\mathbf{e}_0 + x^a \mathbf{e}_a) \wedge (B^{mn} \mathbf{e}_{mn})$$

This produces a trivector (grade-3 in the 5D PGA space).

**Components**:

Mixed (orientation):
$$p^{0mn} = B^{mn}$$

Euclidean (moment):
$$p^{abc} = \frac{1}{3!} x^d B^{mn} \, \varepsilon^{dmnabc}$$

where the antisymmetric symbol has 6 indices total.

**Extraction**:

From $\mathbf{p}$:
- Orientation: $B^{mn} = p^{0mn}$
- Position: Requires solving for the point closest to origin on the plane

### Geometric Interpretation

This decomposition reflects the fundamental bulk-weight structure in projective geometric algebra:

The **weight** (mixed components containing $e_0$) carries the intrinsic geometric character—the orientation of the flat within the ambient space. These components are independent of where the flat is positioned.

The **bulk** (purely Euclidean components) encodes the extrinsic positioning—how the oriented object is offset from the origin. For lines, this is the familiar moment vector. For planes and higher-dimensional flats, this generalizes to higher-order moment structures.

The construction via $\mathbf{q} \wedge \mathbf{p}$ naturally produces this decomposition because the wedge product with $\mathbf{e}_0$ creates the mixed components directly from $\mathbf{p}$, while the wedge product with $x^m \mathbf{e}_m$ generates the moment structure encoding position.

### Dimensional Scaling

The pattern scales systematically across dimensions:

In $d$-dimensional Euclidean space:
- A $k$-flat is represented by a $(k+1)$-blade in PGA
- Mixed components: $\binom{d}{k}$ independent values (the orientation)
- Euclidean components: $\binom{d}{k+1}$ independent values (the moment)
- Total components: $\binom{d+1}{k+1} = \binom{d}{k} + \binom{d}{k+1}$

This confirms that the total degrees of freedom match the expected count: a $k$-flat in $d$-dimensional space has $d \cdot k - \binom{k}{2}$ degrees of freedom (k vectors each with d components, minus the $\binom{k}{2}$ constraints from orthogonalization), plus $d$ degrees for positioning, giving $dk - k(k-1)/2 + d = d(k+1) - k(k-1)/2$ total degrees of freedom.
