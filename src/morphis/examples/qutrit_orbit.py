"""
Qutrit Orbit

A spin-1 in a magnetic field tilted from z by an angle β, with a quadratic
shift D along z:

    H = cos β S_z + sin β S_x + D S_z²

in units of the field strength. Its state space is ℂ³, read in morphis as the
space with three modes, ℝ⁶ with directions 1, 1̇, 2, 2̇, 3, 3̇. The modes are
taken to be the energy eigenstates |k⟩, with energies E_k, so the evolution
is diagonal: each amplitude keeps its modulus a_k and turns at its own rate,

    z^k(t) = a_k e^{-i E_k t},   Σ a_k² = 1

and the state stays on the 3-torus of fixed populations in the unit sphere S⁵.
The overall phase cannot be observed. Measuring the phases from the ground
state's leaves the 2-torus of relative phases, with the ground amplitude a
constant a_1, and the state winds over it along a straight line of the torus
coordinates,

    (u, v) = -(ω_21 t, ω_31 t),   ω_k1 = E_k - E_1

The ratio ω_31 / ω_21 is irrational, so the orbit never closes and fills the
torus densely. Here it is close to 2: the orbit is nearly the (1, 2) curve,
and it precesses slowly around the torus.

The figure drops the constant ground amplitude and draws the rest, a torus on
the 3-sphere of radius ρ = √(a_2² + a_3²) in the directions 2, 2̇, 3, 3̇, by
stereographic projection from the pole ρ e_3̇, where it becomes a torus of
revolution, drawn translucent in pen and stipple: its contour inked and its
body stippled, with no grid, so the orbit shows through it, lighter where it
runs behind. The animation draws the state and the orbit it has traced.

Run:
    uv run python -m morphis.examples.qutrit_orbit [output.png] [ink|parchment|chalkboard]
    uv run python -m morphis.examples.qutrit_orbit --animate [output.gif | output.mp4] [theme]
"""

import sys

from numpy import abs as np_abs, array, cos, diag, exp, linspace, sin, sqrt
from numpy.linalg import eigh
from numpy.typing import NDArray

from morphis.elements import Dot, Vector, basis_vector, metric
from morphis.manifold import Circle, Patch
from morphis.visuals.ink import Camera, Sketch, animate
from morphis.visuals.projective import Projection


# The Hamiltonian, in units of the field strength
TILT = 0.9
QUADRATIC_SHIFT = 0.4

# The state starts in the lab state m = 0
INITIAL_STATE = array([0.0, 1.0, 0.0], dtype=complex)

DURATION = 60.0
ORBIT_SAMPLES = 6001
ANIMATION_SECONDS = 10.0
FRAME_RATE = 30

# Resolution of animation frames; a GIF of stippled ink grows quickly with it
ANIMATION_DPI = 100

# Cells of the mesh the torus's contour and stipple are drawn from, around the axis and around the tube
MESH_SHAPE = (96, 48)

CAMERA = Camera(azimuth=-32.0, elevation=30.0, distance=14.0, fov=17.0)

# Graphite levels for the nearest and farthest strokes of the orbit; the far side shows through the torus
ORBIT_INK = (1.0, 0.3)

SPACE = metric(modes=3)


# =============================================================================
# The Qutrit
# =============================================================================


def hamiltonian() -> NDArray:
    """H = cos β S_z + sin β S_x + D S_z² on the lab states m = +1, 0, -1."""
    spin_z = diag([1.0, 0.0, -1.0])
    spin_x = array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / sqrt(2)

    return cos(TILT) * spin_z + sin(TILT) * spin_x + QUADRATIC_SHIFT * spin_z @ spin_z


ENERGIES, EIGENSTATES = eigh(hamiltonian())

# Moduli of the amplitudes on the energy eigenstates, conserved by the evolution
MODULI = np_abs(EIGENSTATES.conj().T @ INITIAL_STATE)
GAPS = ENERGIES[1:] - ENERGIES[0]


def relative_state(u, v):
    """The state on the torus of relative phases: ground amplitude real, the others turned by u and v."""
    return [MODULI[0] + 0 * u, MODULI[1] * exp(1j * u), MODULI[2] * exp(1j * v)]


TORUS = Patch(relative_state, Circle() * Circle(), SPACE)


def phases(t: NDArray) -> NDArray:
    """The torus coordinates of the state at times t: a straight line, -(ω_21 t, ω_31 t)."""
    return -GAPS * array(t, dtype=float)[..., None]


def evolved(t: NDArray) -> NDArray:
    """The lab-state amplitudes at times t, from ψ(t) = V e^{-iEt} V† ψ(0)."""
    weights = EIGENSTATES.conj().T @ INITIAL_STATE
    turned = exp(-1j * ENERGIES * array(t, dtype=float)[..., None]) * weights

    return turned @ EIGENSTATES.T


# From the 3-sphere of the excited modes, at the pole ρ e_3̇
RADIUS = float(sqrt(MODULI[1] ** 2 + MODULI[2] ** 2))
EXCITED = Projection.orthographic([basis_vector(label, SPACE) for label in (2, Dot(2), 3, Dot(3))])
PROJECTION = Projection.stereographic(RADIUS * basis_vector(4, EXCITED.target)) @ EXCITED


def report() -> None:
    """Check the orbit on the torus against the evolution in the lab basis."""
    times = linspace(0.0, DURATION, 241)
    lab = evolved(times)

    # Read the lab state in the energy eigenbasis, with the ground state's phase removed
    eigen = lab @ EIGENSTATES.conj()
    relative = eigen * (eigen[:, :1].conj() / np_abs(eigen[:, :1]))
    on_torus = Vector.from_amplitudes(relative, SPACE).data
    line = TORUS(phases(times)).data

    print(f"{TORUS!r}")
    print(f"  energies          {', '.join(f'{energy:+.4f}' for energy in ENERGIES)}")
    print(f"  populations       {', '.join(f'{modulus**2:.4f}' for modulus in MODULI)}")
    print(f"  frequency ratio   ω_31 / ω_21 = {GAPS[1] / GAPS[0]:.6f}")
    print(f"  orbit on torus    max |ψ - X(u, v)| = {np_abs(on_torus - line).max():.1e}")
    print(f"  lab populations   m = 0 runs from {abs(lab[:, 1]).min() ** 2:.3f} to {abs(lab[:, 1]).max() ** 2:.3f}")


# =============================================================================
# Figure
# =============================================================================


def strokes(sketch: Sketch, path: NDArray, depths, weight: str, pieces: int, levels) -> None:
    """Ink a drawn curve in short pieces, each darker the nearer it is to the camera."""
    _, depth = sketch.camera.project(path)
    near, far = depths
    bounds = linspace(0, len(path) - 1, pieces + 1).astype(int)
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        nearness = (far - depth[start : stop + 1].mean()) / (far - near)
        level = levels[1] + (levels[0] - levels[1]) * nearness
        sketch.curve(path[start : stop + 1], weight=weight, level=float(level))


def create_sketch(theme: str = "ink", t: float = DURATION) -> Sketch:
    sketch = Sketch(camera=CAMERA, size=(8.0, 6.0), seed=5, theme=theme)

    # The torus of relative phases, translucent: a pen contour and stipple, no grid
    mesh = TORUS.mesh(MESH_SHAPE)
    torus = PROJECTION(mesh.vertices)
    sketch.shell(torus, mesh.cells.cells)
    _, depth = sketch.camera.project(torus)
    depths = (depth.min(), depth.max())

    # The orbit traced so far, and the state now
    count = max(2, int(ORBIT_SAMPLES * t / DURATION))
    orbit = PROJECTION(TORUS(phases(linspace(0.0, t, count))))
    strokes(sketch, orbit, depths, "fine", max(1, count // 25), ORBIT_INK)

    sketch.point(orbit[-1], radius=4.0)
    sketch.label(r"$\mathbf{ψ}$", orbit[-1], offset=(9, 7), size=15)

    return sketch


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--animate"]
    is_animation = "--animate" in sys.argv[1:]
    default = "figures/qutrit-orbit/qutrit-orbit.gif" if is_animation else "figures/qutrit-orbit/qutrit-orbit.png"
    output = arguments[0] if arguments else default
    theme = arguments[1] if len(arguments) > 1 else "ink"

    report()
    if is_animation:
        times = linspace(0.0, DURATION, int(ANIMATION_SECONDS * FRAME_RATE) + 1)[1:]
        path = animate(lambda t: create_sketch(theme, t), times, output, frame_rate=FRAME_RATE, dpi=ANIMATION_DPI)
    else:
        path = create_sketch(theme).save(output)

    print(f"Saved {path}")
