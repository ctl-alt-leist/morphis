"""
Conjugate Planes

A realified state space 𝒱 = ℝ^{2n}: each mode m owns a direction e_m and its
partner e_ṁ, the quarter turn of e_m, and the two span the conjugate plane of
the mode, the blade e_{mṁ}. A state is

    ψ = η^m e_m + ξ^m e_ṁ,   ψ^μ ψ^μ = 2

The figure picks two of the modes, a and b, and the state's exact shadows ψ_a
and ψ_b in their planes. The computation keeps just those two modes, so the
true space here is ℝ⁴ with directions a, ȧ, b, ḃ. The figure then chooses how
to draw this 4D arrangement in three dimensions: both mode directions e_a and
e_b are drawn along the same page direction, and each plane is set out at its
own place in the space, joined to the state by dashed construction lines.

The modes are energy eigenstates, so the evolution is exact and geometric: a
rigid rotation by ω_m t inside each conjugate plane, the sandwich
ψ(t) = R ψ(0) R̃ with the rotor

    R(t) = rotor(e_{aȧ}, ω_a t) · rotor(e_{bḃ}, ω_b t)

The two factors commute because the planes are orthogonal. Each shadow turns
in the right-handed sense of its plane, from e_m toward e_ṁ, keeping its length,
and the drawn state is the depiction of the evolved 4D state at every instant.
The Schrödinger rotor U = e^{H t/2} with H = E_m e_{mṁ} turns the opposite way;
the right-handed sense is chosen here.

In code, e_m is `E_M` and its partner e_ṁ is `F_M`.

Run:
    uv run python -m morphis.examples.conjugate_planes [output.png] [ink|parchment|chalkboard]
    uv run python -m morphis.examples.conjugate_planes --animate [output.mp4] [theme]
"""

import sys
from functools import cache

from numpy import array, concatenate, linspace, pi, searchsorted, sqrt
from numpy.linalg import norm

from morphis.elements import Vector, basis_vectors, euclidean_metric
from morphis.transforms import rotor
from morphis.visuals.ink import Camera, Depiction, OrganicSpace, Sketch, animate


# Mode frequencies ω_m = E_m / ħ; their ratio sets the closed path the state traces
FREQUENCY_A = 1.0
FREQUENCY_B = 2.0
PERIOD = 2 * pi / FREQUENCY_A
ORBIT_TIMES = linspace(0.0, PERIOD, 721)

DURATION = 10.0
FRAME_RATE = 30

# Where the state arrow is based and how large it is drawn: room for its whole
# orbit in the open space between the two planes
STATE_ORIGIN = (0.35, 0.9, -0.35)
STATE_SCALE = 0.8

# True geometry: the two chosen modes, directions (a, ȧ, b, ḃ) as the geometric indices 1..4
METRIC = euclidean_metric(4)
E_A, F_A, E_B, F_B = basis_vectors(METRIC)

# Components (η^a, ξ^a, η^b, ξ^b), normalized to ψ^μ ψ^μ = 2
COMPONENTS = array([0.75, 0.95, 0.5, 0.85])
INITIAL_STATE = Vector(sqrt(2) * COMPONENTS / norm(COMPONENTS), grade=1, metric=METRIC)


def evolve(state: Vector, t: float) -> Vector:
    """Phase evolution for time t: rotate each conjugate plane by ω_m t, from e_m toward e_ṁ."""
    R = rotor(E_A ^ F_A, FREQUENCY_A * t) * rotor(E_B ^ F_B, FREQUENCY_B * t)
    evolved = (R * state * ~R).data[1]

    return evolved


def shadows(state: Vector) -> tuple[Vector, Vector]:
    """Orthogonal projections of the state onto the two conjugate planes: η^m e_m + ξ^m e_ṁ."""
    shadow_a = state.on[1].data.item() * E_A + state.on[2].data.item() * F_A
    shadow_b = state.on[3].data.item() * E_B + state.on[4].data.item() * F_B

    return shadow_a, shadow_b


@cache
def orbit() -> Vector:
    """The evolved state over one period, as a lot of 4D vectors on a fine time grid (computed once)."""
    times = ORBIT_TIMES
    R = rotor(E_A ^ F_A, FREQUENCY_A * times) * rotor(E_B ^ F_B, FREQUENCY_B * times)
    states = (R * INITIAL_STATE * ~R).data[1]

    return states


def create_sketch(theme: str = "ink", t: float = 0.0) -> Sketch:
    state = evolve(INITIAL_STATE, t)
    shadow_a, shadow_b = shadows(state)

    # Depiction: both mode directions e_a and e_b along x; plane a stands upright, plane b lies flat
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
    origin = array(STATE_ORIGIN)
    scale = STATE_SCALE
    anchor_a = array([-2.25, 0.5, 0.75])
    anchor_b = array([1.9, 0.1, -2.25])
    reach = 1.3

    for anchor, e, f, mode in ((anchor_a, E_A, F_A, "a"), (anchor_b, E_B, F_B, "b")):
        corner = anchor - reach * (sketch.drawn(e) + sketch.drawn(f))
        sketch.plane(e, f, at=corner, span=((0.0, 2 * reach), (0.0, 2 * reach)), grid=6, tone=0.3)
        sketch.vector(0.5 * e, at=corner + 0.75 * sketch.drawn(e), weight="fine", head=0.8)
        sketch.vector(0.5 * f, at=corner + 0.75 * sketch.drawn(f), weight="fine", head=0.8)
        sketch.label(r"$\mathbf{e}_{" + mode + "}$", corner + 1.0 * sketch.drawn(e), offset=(3, -13), size=14)
        sketch.label(
            r"$\mathbf{e}_{\overset{\bullet}{" + mode + "}}$", corner + 1.0 * sketch.drawn(f), offset=(-17, 3), size=14
        )

    # Phase circles through each shadow, arrowed in the right-handed sense of the evolution
    sketch.circle(anchor_a, E_A, F_A, radius=norm(shadow_a.data), arc=(1.0, 2.6 + pi), arrow=True)
    sketch.circle(anchor_b, E_B, F_B, radius=norm(shadow_b.data), arc=(1.25, 2.85 + pi), arrow=True)

    # The drawn state's closed orbit over one period, and the part already traversed
    path = origin + scale * sketch.drawn(orbit())
    sketch.curve(path, dashed=True, weight="hair", level=0.35)
    elapsed = searchsorted(ORBIT_TIMES, t % PERIOD)
    traversed = concatenate([path[:elapsed], [origin + scale * sketch.drawn(state)]])
    if len(traversed) > 1:
        sketch.curve(traversed, level=0.5)

    tip_a = sketch.vector(
        shadow_a, at=anchor_a, weight="bold", label=r"$\mathbf{ψ}_a$", label_offset=-13, label_size=14
    )
    tip_b = sketch.vector(
        shadow_b, at=anchor_b, weight="bold", label=r"$\mathbf{ψ}_b$", label_offset=-17, label_size=14
    )
    sketch.point(anchor_a)
    sketch.point(anchor_b)

    tip = sketch.vector(
        state, at=origin, scale=scale, weight="heavy", label=r"$\mathbf{ψ}$", label_offset=16, label_size=14
    )
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
        path = animate(lambda t: create_sketch(theme, t), times, output, frame_rate=FRAME_RATE)
    else:
        path = create_sketch(theme).save(output)

    print(f"Saved {path}")
