"""
A Calabi–Yau Slice

The quintic threefold z_1^5 + ... + z_5^5 = 0 in ℂℙ⁴ is the standard
Calabi–Yau manifold, and its standard picture, after Hanson, is a slice of it:
the complex curve

    z_1^5 + z_2^5 = 1

in ℂ², a real surface in ℝ⁴. In morphis ℂ² is the space with two modes, with
directions 1, 1̇, 2, 2̇, and a point of the curve is written by its amplitudes.

Since cosh²w - sinh²w = 1, the curve is covered by the 25 patches

    z_1 = e^{2πi k_1 / 5} cosh(w)^{2/5},    z_2 = e^{2πi k_2 / 5} (-i sinh w)^{2/5}

over the complex coordinate w ∈ [-a, a] + i [0, π/2], one for each pair of
fifth roots of unity. Each patch is holomorphic in w, so its tangent planes are
complex lines (Kähler angle 0) and the surface is minimal (k = 0). Welded at
coinciding samples, the patches form a surface of genus 6 with its 5 points at
infinity cut off, χ = 2 - 2·6 - 5 = -15, with 5 boundary circles.

The picture is the orthographic projection onto e_1, e_2, and cos α e_1̇ +
sin α e_2̇. It hides the direction n = -sin α e_1̇ + cos α e_2̇ and pinches the
surface wherever a tangent plane contains n. A tangent line of the curve is
spanned by (z_2⁴, -z_1⁴) over ℂ, and it contains n = (-i sin α, i cos α)
exactly when (z_2 / z_1)⁴ = tan α. So the pinch points are z_2 = t z_1 with
t⁴ = tan α and z_1⁵ = 1 / (1 + t⁵); at α = π/4 the root t = -1 sends the
point to infinity, leaving 3 × 5 = 15.

The figures build the surface in stages:

    patch     one patch beside its coordinate rectangle
    phases    the five patches with k_1 = 0, fanned around z_2 = 0
    atlas     all 25 patches, colored by k_1
    pinch     the welded surface shaded by how nearly the projection collapses
              each tangent plane, with the 15 pinch points

Run:
    uv run python -m morphis.examples.calabi_yau [stage | all] [ink|parchment|chalkboard] [folder]
"""

import sys

from numpy import abs as np_abs, array, cos, cosh, exp, pi, sin, sinh, stack, tan
from numpy.linalg import norm

from morphis.elements import Dot, Frame, Vector, basis_vector, metric
from morphis.manifold import Atlas, Interval, Mesh, Patch
from morphis.operations import kahler_angle
from morphis.visuals.ink import Camera, Sketch
from morphis.visuals.projective import Projection


DEGREE = 5

# Reach of Re w: how far toward the points at infinity each patch extends
REACH = 1.2

# Samples per patch along Re w and Im w
GRID = (33, 17)

# Projection angle mixing the two imaginary directions into the third drawing axis
ANGLE = pi / 4

CAMERA = Camera(azimuth=-35.0, elevation=25.0, distance=14.0, fov=15.0)

# Cells whose drawn shape is squeezed below this ratio s_min / s_max are tinted toward red
COLLAPSE_SHADED = 0.25

# One accent per value of k
PALETTE = ("blue", "red", "green", "ochre", "violet")

STAGES = ("patch", "phases", "atlas", "pinch")

SPACE = metric(modes=2)
E_1, F_1, E_2, F_2 = (basis_vector(label, SPACE) for label in (1, Dot(1), 2, Dot(2)))

PROJECTION = Projection.orthographic([E_1, E_2, cos(ANGLE) * F_1 + sin(ANGLE) * F_2])


def sheet(k1: int, k2: int) -> Patch:
    """The patch of the curve with phases e^{2πi k_1 / 5} and e^{2πi k_2 / 5}."""

    def X(w):
        z1 = exp(2j * pi * k1 / DEGREE) * cosh(w) ** (2 / DEGREE)
        z2 = exp(2j * pi * k2 / DEGREE) * (-1j * sinh(w)) ** (2 / DEGREE)

        return [z1, z2]

    return Patch(X, Interval(-REACH, REACH) + 1j * Interval(0.0, pi / 2), SPACE)


PAIRS = [(k1, k2) for k1 in range(DEGREE) for k2 in range(DEGREE)]
ATLAS = Atlas([sheet(k1, k2) for k1, k2 in PAIRS])


# =============================================================================
# Geometry
# =============================================================================


def pinch_points() -> Vector:
    """The points where the projection's hidden direction lies in the tangent plane."""
    points = []
    for m in range(4):
        t = tan(ANGLE) ** 0.25 * 1j**m
        if np_abs(1 + t**DEGREE) < 1e-9:
            continue

        for k in range(DEGREE):
            z1 = (1 / (1 + t**DEGREE)) ** (1 / DEGREE) * exp(2j * pi * k / DEGREE)
            points.append([z1, t * z1])

    return Vector.from_amplitudes(array(points), SPACE)


def tangent_frame(points: Vector) -> Frame:
    """The tangent plane of the curve at its points: u = (z_2⁴, -z_1⁴) and J u."""
    z1, z2 = points.amplitudes.T
    u = Vector.from_amplitudes(stack([z2**4, -(z1**4)], axis=-1), SPACE)
    Ju = Vector.from_amplitudes(stack([1j * z2**4, -1j * z1**4], axis=-1), SPACE)

    return Frame(data=stack([u.data, Ju.data], axis=-2), metric=SPACE)


def cell_centers(patch: Patch):
    """Coordinates at the centers of a patch's grid cells, in the order of its cell complex."""
    grid = patch.grid(GRID)

    return 0.25 * (grid[:-1, :-1] + grid[1:, :-1] + grid[:-1, 1:] + grid[1:, 1:])


def report(mesh: Mesh) -> None:
    sigma = cell_centers(ATLAS.patches[0])
    patch = ATLAS.patches[0]
    angles = kahler_angle(patch.tangent_frame(sigma))
    mean = norm(patch.mean_curvature(sigma).data, axis=-1)
    curvature = patch.gaussian_curvature(sigma)
    z = mesh.vertices.amplitudes
    pinches = pinch_points()
    collapse = PROJECTION.pinching(pinches, tangent_frame(pinches))

    print(f"{ATLAS!r}")
    print(f"  on the curve      max |z_1⁵ + z_2⁵ - 1| = {np_abs(z[:, 0] ** 5 + z[:, 1] ** 5 - 1).max():.1e}")
    print(f"  complex line      max θ_K = {angles.max():.1e}")
    print(f"  minimal           max |k| = {mean.max():.1e}")
    print(f"  curvature         K from {curvature.min():.3f} to {curvature.max():.3f}")
    chi, n_boundaries = mesh.cells.euler_characteristic(), mesh.cells.boundary().components()
    print(f"  topology          χ = {chi}, {n_boundaries} boundary circles")
    print(f"  pinch points      {len(pinches.data)}, max s_min / s_max = {collapse.max():.1e}")


# =============================================================================
# Figures
# =============================================================================


def stage_patch(sketch: Sketch) -> None:
    """One patch, k_1 = k_2 = 0, beside the rectangle of w that it is a map of."""
    patch = ATLAS.patches[0]
    mesh = patch.mesh(GRID)
    sketch.surface(PROJECTION(mesh.vertices), mesh.cells.cells, colors="blue")

    # The coordinate rectangle, laid flat to the left and scaled to the patch
    sigma = patch.grid(GRID).reshape(-1, 2)
    scale, (x, y, z) = 0.45, (-2.2, 0.0, -0.9)
    flat = stack([scale * sigma[:, 0] + x, scale * sigma[:, 1] + y, 0 * sigma[:, 0] + z], axis=-1)
    sketch.surface(flat, mesh.cells.cells, colors="sepia")

    sketch.label(r"$w$", (x, y - 0.25, z), offset=(0, -8), size=16)
    sketch.label(r"$\mathbf{X}(w)$", (1.3, 1.3, 0.5), offset=(18, 12), size=16)
    sketch.line((x + 0.75, y + 0.35, z + 0.1), (-0.1, 0.3, 0.0), dashed=True, weight="fine")


def stage_phases(sketch: Sketch) -> None:
    """The five patches with k_1 = 0: z_2 runs over all its fifth-root phases."""
    patches = [ATLAS.patches[PAIRS.index((0, k2))] for k2 in range(DEGREE)]
    mesh = Mesh.union(*(patch.mesh(GRID) for patch in patches)).welded(1e-6)
    colors = [PALETTE[int(piece)] for piece in mesh.pieces]
    sketch.surface(PROJECTION(mesh.vertices), mesh.cells.cells, colors=colors)


def stage_atlas(sketch: Sketch, mesh: Mesh) -> None:
    """All 25 patches, each colored by k_1."""
    colors = [PALETTE[PAIRS[int(piece)][0]] for piece in mesh.pieces]
    sketch.surface(PROJECTION(mesh.vertices), mesh.cells.cells, colors=colors)


def stage_pinch(sketch: Sketch, mesh: Mesh) -> None:
    """The welded surface shaded by s_min / s_max of the projection on each cell, red where it collapses."""
    collapse = []
    for patch in ATLAS.patches:
        sigma = cell_centers(patch)
        collapse.append(PROJECTION.pinching(patch(sigma), patch.tangent_frame(sigma)).ravel())

    values = array([value for block in collapse for value in block])
    graphite = array(sketch.theme.gray(0.25))
    red = array(sketch.theme.color("red"))
    weight = (1 - values / COLLAPSE_SHADED).clip(0, 1) ** 0.6
    colors = [tuple(c) for c in graphite + (red - graphite) * weight[:, None]]
    sketch.surface(
        PROJECTION(mesh.vertices), mesh.cells.cells, colors=colors, edge_level=0.45, dots=PROJECTION(pinch_points())
    )


def create_sketch(stage: str, theme: str = "ink", mesh: Mesh | None = None) -> Sketch:
    sketch = Sketch(camera=CAMERA, size=(8.0, 8.0) if stage != "patch" else (10.0, 6.0), seed=5, theme=theme)
    mesh = mesh if mesh is not None else ATLAS.mesh(GRID, tolerance=1e-6)

    if stage == "patch":
        stage_patch(sketch)

    elif stage == "phases":
        stage_phases(sketch)

    elif stage == "atlas":
        stage_atlas(sketch, mesh)

    else:
        stage_pinch(sketch, mesh)

    return sketch


if __name__ == "__main__":
    chosen = sys.argv[1] if len(sys.argv) > 1 else "all"
    theme = sys.argv[2] if len(sys.argv) > 2 else "ink"
    folder = sys.argv[3] if len(sys.argv) > 3 else "figures/higher-dimensional"
    stages = STAGES if chosen == "all" else (chosen,)

    mesh = ATLAS.mesh(GRID, tolerance=1e-6)
    report(mesh)

    for stage in stages:
        path = create_sketch(stage, theme, mesh).save(f"{folder}/calabi-yau-{stage}.png")
        print(f"Saved {path}")
