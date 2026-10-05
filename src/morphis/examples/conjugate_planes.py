"""
Conjugate Planes

A qubit state realified: ℂ² becomes ℝ⁴ with directions 1, 1̇, 2, 2̇, paired
into two conjugate planes, one per mode. The state ψ splits exactly into its
shadows ψ_a and ψ_b in the two planes. The figure then chooses how to draw
this 4D arrangement in three dimensions: both representative directions are
drawn along the same page direction, and each plane is set out at its own
place in the space, joined to the state by dashed construction lines.

The modes are energy eigenstates, so the evolution is exact and geometric:
each mode's phase turns as e^{iω_m t}. Multiplication by i turns f toward g,
so this is a rigid, right-handed rotation by +ω_m t inside conjugate plane m
(the orientation f_m∧g_m), and the whole evolution is the sandwich
ψ(t) = R ψ(0) R̃ with the rotor

    R(t) = rotor(f_a∧g_a, ω_a t) · rotor(f_b∧g_b, ω_b t)

The two factors commute because the planes are orthogonal. The shadows keep
their lengths and turn at their own frequencies, and the drawn state is the
depiction of the evolved 4D state at every instant. (The Schrödinger phase
e^{-iEt/ħ} turns the opposite way; the right-handed sense is chosen here.)

Run:
    uv run python -m morphis.examples.conjugate_planes [output.png] [ink|parchment|chalkboard]
    uv run python -m morphis.examples.conjugate_planes --animate [output.mp4] [theme]
"""

import sys
from functools import cache

from numpy import array, concatenate, linspace, pi, searchsorted
from numpy.linalg import norm

from morphis.elements import Vector, basis_vectors, euclidean_metric
from morphis.transforms import rotor
from morphis.visuals.ink import Camera, Depiction, OrganicSpace, Sketch, animate


# Mode frequencies ω_m = E_m / ħ; their ratio sets the closed path the state traces
OMEGA_A = 1.0
OMEGA_B = 2.0
PERIOD = 2 * pi / OMEGA_A
ORBIT_TIMES = linspace(0.0, PERIOD, 721)

DURATION = 10.0

# Where the state arrow is based and how large it is drawn: a still can lean the
# arrow between the planes, while the moving state needs room for its whole orbit
STILL_ORIGIN = (-1.15, -0.9, -1.1)
STILL_SCALE = 1.55
MOTION_ORIGIN = (0.35, 0.9, -0.35)
MOTION_SCALE = 0.8
FRAME_RATE = 30

# True geometry: the realified qubit, directions (1, 1̇, 2, 2̇) as e_1..e_4
METRIC = euclidean_metric(4)
F_A, G_A, F_B, G_B = basis_vectors(METRIC)
INITIAL_STATE = Vector([0.75, 0.95, 0.5, 0.85], grade=1, metric=METRIC)


def evolve(state: Vector, t: float) -> Vector:
    """Phase evolution for time t: rotate each conjugate plane by +ω_m t, right-handed about f_m∧g_m."""
    R = rotor(F_A ^ G_A, OMEGA_A * t) * rotor(F_B ^ G_B, OMEGA_B * t)
    evolved = (R * state * ~R).data[1]

    return evolved


def shadows(state: Vector) -> tuple[Vector, Vector]:
    """Orthogonal projections of the state onto the two conjugate planes."""
    a = state.on[1].data.item() * F_A + state.on[2].data.item() * G_A
    b = state.on[3].data.item() * F_B + state.on[4].data.item() * G_B

    return a, b


@cache
def orbit() -> Vector:
    """The evolved state over one period, as a lot of 4D vectors on a fine time grid (computed once)."""
    times = ORBIT_TIMES
    R = rotor(F_A ^ G_A, OMEGA_A * times) * rotor(F_B ^ G_B, OMEGA_B * times)
    states = (R * INITIAL_STATE * ~R).data[1]

    return states


def create_sketch(
    theme: str = "ink",
    t: float = 0.0,
    trail: bool = False,
    origin: tuple[float, float, float] = STILL_ORIGIN,
    scale: float = STILL_SCALE,
) -> Sketch:
    state = evolve(INITIAL_STATE, t)
    shadow_a, shadow_b = shadows(state)

    # Depiction: both representative directions along x; plane a stands upright, plane b lies flat
    depiction = Depiction(METRIC, {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})

    sketch = Sketch(
        camera=Camera(azimuth=-24.0, elevation=34.0, distance=16.0, fov=16.0),
        depiction=depiction,
        size=(8.0, 6.0),
        seed=4,
        theme=theme,
    )
    sketch.space(OrganicSpace(seed=3, stretch=(1.6, 1.25, 1.05), size=3.2))

    # Placement: where each piece sits in the drawing space
    origin = array(origin)
    anchor_a = array([-2.25, 0.5, 0.75])
    anchor_b = array([1.9, 0.1, -2.25])
    reach = 1.3

    for anchor, u, v, mode in ((anchor_a, F_A, G_A, "a"), (anchor_b, F_B, G_B, "b")):
        corner = anchor - reach * (sketch.drawn(u) + sketch.drawn(v))
        sketch.plane(u, v, at=corner, span=((0.0, 2 * reach), (0.0, 2 * reach)), grid=6, tone=0.3)
        sketch.vector(0.5 * u, at=corner + 0.75 * sketch.drawn(u), weight="fine", head=0.8)
        sketch.vector(0.5 * v, at=corner + 0.75 * sketch.drawn(v), weight="fine", head=0.8)
        sketch.label(rf"$f_{mode}$", corner + 1.0 * sketch.drawn(u), offset=(3, -13), size=14)
        sketch.label(rf"$g_{mode}$", corner + 1.0 * sketch.drawn(v), offset=(-12, 3), size=14)

    # Phase circles through each shadow, arrowed in the right-handed sense of the evolution
    sketch.circle(anchor_a, F_A, G_A, radius=norm(shadow_a.data), arc=(1.0, 2.6 + pi), arrow=True)
    sketch.circle(anchor_b, F_B, G_B, radius=norm(shadow_b.data), arc=(1.25, 2.85 + pi), arrow=True)

    if trail:
        # The drawn state's closed orbit over one period, and the part already traversed
        path = origin + scale * sketch.drawn(orbit())
        sketch.curve(path, dashed=True, weight="hair", level=0.35)
        elapsed = searchsorted(ORBIT_TIMES, t % PERIOD)
        traversed = concatenate([path[:elapsed], [origin + scale * sketch.drawn(state)]])
        if len(traversed) > 1:
            sketch.curve(traversed, level=0.5)

    tip_a = sketch.vector(shadow_a, at=anchor_a, weight="bold", label="$ψ_a$", label_offset=-13, label_size=14)
    tip_b = sketch.vector(shadow_b, at=anchor_b, weight="bold", label="$ψ_b$", label_offset=-17, label_size=14)
    sketch.point(anchor_a)
    sketch.point(anchor_b)

    tip = sketch.vector(state, at=origin, scale=scale, weight="heavy", label="$ψ$", label_offset=16, label_size=14)
    sketch.point(origin, radius=4.0)

    for start, end in ((origin, anchor_a), (origin, anchor_b), (tip, tip_a), (tip, tip_b)):
        sketch.line(start, end)

    sketch.label(r"$\mathcal{V}$", array([2.35, 0.4, 1.95]), offset=(0, 0), size=24)

    return sketch


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--animate"]
    is_animation = "--animate" in sys.argv[1:]
    default = (
        "figures/conjugate-planes/conjugate-planes.mp4"
        if is_animation
        else "figures/conjugate-planes/conjugate-planes.png"
    )
    output = arguments[0] if arguments else default
    theme = arguments[1] if len(arguments) > 1 else "ink"

    if is_animation:
        times = linspace(0.0, PERIOD, int(DURATION * FRAME_RATE), endpoint=False)
        path = animate(
            lambda t: create_sketch(theme, t, trail=True, origin=MOTION_ORIGIN, scale=MOTION_SCALE),
            times,
            output,
            fps=FRAME_RATE,
        )
    else:
        path = create_sketch(theme).save(output)

    print(f"Saved {path}")
