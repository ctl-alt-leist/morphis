"""
Clifford Torus

A qubit state is a unit vector of ℂ², read in morphis as the space with two
modes, ℝ⁴ with directions 1, 1̇, 2, 2̇. Fixing the populations |z^1|² = cos²(χ/2)
and |z^2|² = sin²(χ/2) leaves the two phases free, and the states sweep out a
torus in the unit sphere S³:

    X(θ, φ) = (cos(χ/2) e^{iθ}, sin(χ/2) e^{iφ})

At χ = π/2 this is the Clifford torus. The script checks its geometry against
the closed forms, all computed in the ambient ℝ⁴ from the patch alone:

    h_ab = ½ δ_ab           flat, K = 0, though bent in ℝ⁴
    k = -X                  mean curvature of a minimal surface of S³
    θ_K = π/2               Lagrangian: J carries each tangent plane to the normal plane

The global phase e^{iα} ψ moves θ and φ together, so each phase orbit is a
(1, 1) curve on the torus, a fiber of the Hopf map.

The figure draws the torus by stereographic projection from the pole e_2̇ of
S³ into ℝ³, where it becomes a torus of revolution. Coordinate circles are
inked darker in front and lighter behind; one phase orbit is drawn bold.

Run:
    uv run python -m morphis.examples.clifford_torus [output.png] [ink|parchment|chalkboard]
"""

import sys

from numpy import abs as np_abs, cos, exp, linspace, newaxis, pi, sin, stack
from numpy.typing import NDArray

from morphis.elements import Dot, metric
from morphis.manifold import Circle, Patch
from morphis.operations import kahler_angle
from morphis.visuals.ink import Camera, Sketch


# Populations fixed by the polar angle χ on the Bloch sphere; π/2 is the Clifford torus
POLAR_ANGLE = pi / 2

# Coordinate circles drawn in each direction, and the samples along each
N_CIRCLES = (28, 14)
N_SAMPLES = 241

# The phase orbit drawn bold starts at these phases
ORBIT_START = (0.0, 0.45 * pi)
ORBIT_LABEL_AT = 200

CAMERA = Camera(azimuth=-30.0, elevation=28.0, distance=14.0, fov=17.0)

# Half-length of the drawn axis of revolution
AXIS_REACH = 2.4

# Graphite levels for the nearest and farthest strokes
INK_RANGE = (0.95, 0.22)

SPACE = metric(modes=2)


def state(θ, φ):
    """Mode amplitudes of the state with phases θ and φ at fixed populations."""
    return [cos(POLAR_ANGLE / 2) * exp(1j * θ), sin(POLAR_ANGLE / 2) * exp(1j * φ)]


TORUS = Patch(state, Circle() * Circle(), SPACE)


def stereographic(points: NDArray) -> NDArray:
    """Project unit vectors of ℝ⁴ from the pole e_2̇ onto the hyperplane through the origin perpendicular to it."""
    pole = SPACE.to_internal(Dot(2))
    rest = [SPACE.to_internal(label) for label in (1, Dot(1), 2)]

    return points[..., rest] / (1.0 - points[..., pole])[..., newaxis]


# =============================================================================
# Geometry
# =============================================================================


def report() -> None:
    """Check the torus against its closed forms on a grid."""
    sigma = TORUS.grid((24, 24))
    h = TORUS.induced_metric(sigma)
    K = TORUS.gaussian_curvature(sigma)
    mean = TORUS.mean_curvature(sigma).data + TORUS(sigma).data
    angle = kahler_angle(TORUS.tangent_frame(sigma))

    print(f"{TORUS!r}")
    print(f"  induced metric    h_11 = {h[..., 0, 0].mean():.6f}, h_12 = {np_abs(h[..., 0, 1]).max():.1e}")
    print(f"  Gauss curvature   max |K| = {np_abs(K).max():.1e}")
    print(f"  mean curvature    max |k + X| = {np_abs(mean).max():.1e}")
    print(f"  Kähler angle      θ_K / π = {angle.mean() / pi:.6f}")


# =============================================================================
# Figure
# =============================================================================


def depth_range(sketch: Sketch) -> tuple[float, float]:
    """Nearest and farthest camera depth over the whole drawn torus."""
    _, depth = sketch.camera.project(stereographic(TORUS(TORUS.grid(64)).data).reshape(-1, 3))

    return depth.min(), depth.max()


def strokes(sketch: Sketch, path: NDArray, depths, weight: str, pieces: int, levels=INK_RANGE) -> None:
    """Ink a drawn curve in short pieces, each darker the nearer it is to the camera."""
    _, depth = sketch.camera.project(path)
    near, far = depths
    bounds = linspace(0, len(path) - 1, pieces + 1).astype(int)
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        piece = path[start : stop + 1]
        nearness = (far - depth[start : stop + 1].mean()) / (far - near)
        level = levels[1] + (levels[0] - levels[1]) * nearness
        sketch.curve(piece, weight=weight, level=float(level))


def create_sketch(theme: str = "ink") -> Sketch:
    sketch = Sketch(camera=CAMERA, size=(8.0, 6.0), seed=7, theme=theme)
    loop = linspace(0.0, 2 * pi, N_SAMPLES)
    depths = depth_range(sketch)

    # Circles of constant φ (θ runs) and of constant θ (φ runs)
    for fixed in linspace(0.0, 2 * pi, N_CIRCLES[0], endpoint=False):
        sigma = stack([loop, 0 * loop + fixed], axis=-1)
        strokes(sketch, stereographic(TORUS(sigma).data), depths, "hair", 40)

    for fixed in linspace(0.0, 2 * pi, N_CIRCLES[1], endpoint=False):
        sigma = stack([0 * loop + fixed, loop], axis=-1)
        strokes(sketch, stereographic(TORUS(sigma).data), depths, "hair", 20)

    # A global phase orbit e^{iα} ψ: both phases advance together
    θ0, φ0 = ORBIT_START
    orbit = stereographic(TORUS(stack([θ0 + loop, φ0 + loop], axis=-1)).data)
    strokes(sketch, orbit, depths, "bold", 60, levels=(1.0, 0.5))
    sketch.point(orbit[0], radius=3.6)
    sketch.label(r"$\mathbf{ψ}$", orbit[0], offset=(10, 10), size=15)
    sketch.label(r"$e^{iα} \mathbf{ψ}$", orbit[ORBIT_LABEL_AT], offset=(28, -10), size=14)

    # The axis of revolution: the image of the great circle z^1 = 0, through the pole
    sketch.line((0.0, 0.0, -AXIS_REACH), (0.0, 0.0, AXIS_REACH), weight="hair")

    return sketch


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "figures/higher-dimensional/clifford-torus.png"
    theme = sys.argv[2] if len(sys.argv) > 2 else "ink"

    report()
    path = create_sketch(theme).save(output)

    print(f"Saved {path}")
