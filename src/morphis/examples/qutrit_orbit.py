"""
Qutrit Evolution

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
body stippled, with no grid. It is translucent rather than transparent: its
own far side and the stretches of orbit behind the tube are veiled. The
animation draws the state and the orbit it has traced.

A second figure draws the state itself, global phase and all, as the
conjugate-planes figure draws a qubit, without the planes. In ℝ⁶ the state is

    ψ(t) = Σ_k Re(c_k e^{-i E_k t}) e_k + Im(c_k e^{-i E_k t}) e_k̇

a rotation at rate E_k in each conjugate plane at once, with c_k its
amplitude on |k⟩ at t = 0. It is drawn through the linear projection that
treats the six directions alike: each basis direction is sent to one of the
six diagonals of an icosahedron, scaled by 1/√2. The diagonals are
equiangular lines, at arccos(1/√5) to one another, and form a tight frame,
so the map is an orthogonal projection of ℝ⁶ onto a 3-space that favors no
direction. Each mode's circle is drawn as an ellipse traced at its own rate,
and the trajectory is the sum of the three: the pattern it makes is the three
energies beating against one another. The six drawn directions are set out
beside the state as a key.

Run:
    uv run python -m morphis.examples.qutrit_orbit [torus | trajectory | all] [ink|parchment|chalkboard] [--animate]

Figures are written to figures/qutrit-evolution/.
"""

import sys
from functools import partial

from numpy import abs as np_abs, array, cos, diag, exp, linspace, sin, sqrt
from numpy.linalg import eigh
from numpy.typing import NDArray

from morphis.elements import Dot, Vector, basis_vector, metric
from morphis.manifold import Circle, Patch
from morphis.visuals.ink import Camera, OrganicSpace, Sketch, animate
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

FOLDER = "figures/qutrit-evolution"

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


def create_torus_sketch(theme: str = "ink", t: float = DURATION) -> Sketch:
    """The orbit on the torus of relative phases, traced up to time t."""
    sketch = Sketch(camera=CAMERA, size=(8.0, 6.0), seed=5, theme=theme)

    # The orbit traced so far, and the state now
    count = max(2, int(ORBIT_SAMPLES * t / DURATION))
    orbit = PROJECTION(TORUS(phases(linspace(0.0, t, count))))

    # The torus of relative phases, translucent in pen and stipple; it veils the orbit where it runs behind
    mesh = TORUS.mesh(MESH_SHAPE)
    sketch.shell(PROJECTION(mesh.vertices), mesh.cells.cells, curves=[orbit], dots=orbit[-1:], labels=[r"$\mathbf{ψ}$"])

    return sketch


# =============================================================================
# Figure: the State Itself
# =============================================================================


# The six diagonals of an icosahedron: equiangular lines, at arccos(1/√5) to one another. Their signs
# and the camera are chosen so the six drawn directions also spread evenly on the page.
GOLDEN = (1 + sqrt(5)) / 2
DIAGONALS = array([
    [0, 1, GOLDEN],
    [GOLDEN, 0, 1],
    [0, 1, -GOLDEN],
    [-GOLDEN, 0, 1],
    [1, GOLDEN, 0],
    [-1, GOLDEN, 0],
]) / sqrt(1 + GOLDEN**2)

# Each basis direction drawn along its own diagonal; the 1/√2 makes the map an orthogonal projection
LABELS = (1, Dot(1), 2, Dot(2), 3, Dot(3))
EQUIANGULAR = Projection.linear(
    SPACE, {label: tuple(image / sqrt(2)) for label, image in zip(LABELS, DIAGONALS, strict=True)}
)

TRAJECTORY_DURATION = 50.0

# The still ends where the drawn state lies broadside to the page
TRAJECTORY_STILL = 47.14
TRAJECTORY_SAMPLES = 7501
TRAJECTORY_CAMERA = Camera(azimuth=180.0, elevation=20.0, distance=14.0, fov=16.0)

# Where the state is drawn from and how large, and where the key of the six directions is set out
STATE_SCALE = 2.3
KEY_ORIGIN = (2.45, 0.0, -1.0)
KEY_LENGTH = 0.7

# Graphite levels for the nearest and farthest stretches of the trajectory
TRAJECTORY_INK = (1.0, 0.25)


def evolved_state(t: NDArray) -> Vector:
    """ψ(t) in ℝ⁶, global phase and all: each amplitude on |k⟩ turned by -E_k t."""
    amplitudes = EIGENSTATES.conj().T @ INITIAL_STATE
    turned = exp(-1j * ENERGIES * array(t, dtype=float)[..., None]) * amplitudes

    return Vector.from_amplitudes(turned, SPACE)


def create_trajectory_sketch(theme: str = "ink", t: float = TRAJECTORY_STILL) -> Sketch:
    """The state ψ(t) and the path it has traced up to time t, through the equiangular projection."""
    sketch = Sketch(camera=TRAJECTORY_CAMERA, projection=EQUIANGULAR, size=(8.0, 6.0), seed=6, theme=theme)
    sketch.space(OrganicSpace(seed=3, stretch=(1.45, 1.15, 1.1), size=2.45))

    # The path traced so far, darker the nearer it is
    count = max(2, int(TRAJECTORY_SAMPLES * t / TRAJECTORY_DURATION))
    path = STATE_SCALE * sketch.drawn(evolved_state(linspace(0.0, t, count)))
    whole = STATE_SCALE * sketch.drawn(evolved_state(linspace(0.0, TRAJECTORY_DURATION, TRAJECTORY_SAMPLES)))
    _, depth = sketch.camera.project(whole)
    near, far = depth.min(), depth.max()

    _, path_depth = sketch.camera.project(path)
    bounds = linspace(0, len(path) - 1, max(2, count // 18)).astype(int)
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        nearness = (far - path_depth[start : stop + 1].mean()) / (far - near)
        level = TRAJECTORY_INK[1] + (TRAJECTORY_INK[0] - TRAJECTORY_INK[1]) * nearness
        sketch.curve(path[start : stop + 1], weight="fine", level=float(level))

    state = evolved_state(t)
    sketch.vector(state, scale=STATE_SCALE, weight="heavy", label=r"$\mathbf{ψ}$", label_offset=14, label_size=15)
    sketch.point((0.0, 0.0, 0.0), radius=3.6)

    # The key: the six basis directions as drawn, each at the same length and the same angle to the others
    key = array(KEY_ORIGIN)
    for label in LABELS:
        name = r"\overset{\bullet}{" + str(label.mode) + "}" if isinstance(label, Dot) else str(label)
        tip = sketch.vector(KEY_LENGTH * basis_vector(label, SPACE), at=key, weight="fine", head=0.7)
        reach = tip - key
        sketch.label(r"$\mathbf{e}_{" + name + "}$", key + 1.3 * reach, offset=(0, 0), size=12)

    sketch.point(key, radius=2.4)
    sketch.label(r"$\mathcal{V}$", (-2.9, 0.0, 1.85), offset=(0, 0), size=24)

    return sketch


FIGURES = {
    "torus": (create_torus_sketch, DURATION, "qutrit-relative-phases"),
    "trajectory": (create_trajectory_sketch, TRAJECTORY_DURATION, "qutrit-trajectory"),
}


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--animate"]
    is_animation = "--animate" in sys.argv[1:]
    choice = arguments[0] if arguments else "all"
    theme = arguments[1] if len(arguments) > 1 else "ink"
    chosen = list(FIGURES) if choice == "all" else [choice]

    report()
    for name in chosen:
        create, duration, stem = FIGURES[name]
        if is_animation:
            times = linspace(0.0, duration, int(ANIMATION_SECONDS * FRAME_RATE) + 1)[1:]
            output = f"{FOLDER}/{stem}.gif"
            path = animate(partial(create, theme), times, output, frame_rate=FRAME_RATE, dpi=ANIMATION_DPI)
        else:
            path = create(theme).save(f"{FOLDER}/{stem}.png")

        print(f"Saved {path}")
