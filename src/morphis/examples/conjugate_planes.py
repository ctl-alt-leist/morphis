"""
Conjugate Planes

A qubit state realified: ℂ² becomes 𝒱 = ℝ⁴ with directions 1, 1̇, 2, 2̇. Each
mode m owns a direction e_m and its partner e_ṁ, the quarter turn of e_m, and
the two span the conjugate plane of the mode, the blade e_{mṁ}. A state

    ψ = η^m e_m + ξ^m e_ṁ,   ψ^μ ψ^μ = 2

splits exactly into its shadows ψ_1 and ψ_2 in the two planes. The figure then
chooses how to draw this 4D arrangement in three dimensions: both mode
directions e_1 and e_2 are drawn along the same page direction, and each plane
is set out at its own place in the space, joined to the state by dashed
construction lines.

The modes are energy eigenstates, so the evolution is exact and geometric: a
rigid rotation by ω_m t inside each conjugate plane, the sandwich
ψ(t) = R ψ(0) R̃ with the rotor

    R(t) = rotor(e_{11̇}, ω_1 t) · rotor(e_{22̇}, ω_2 t)

The two factors commute because the planes are orthogonal. Each shadow turns
in the right-handed sense of its plane, from e_m toward e_ṁ, keeping its length,
and the drawn state is the depiction of the evolved 4D state at every instant.
The Schrödinger rotor U = e^{H t/2} with H = E_m e_{mṁ} turns the opposite way;
the right-handed sense is chosen here.

In code, e_m is `e_m` and its partner e_ṁ is `f_m`.

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
FREQUENCY_1 = 1.0
FREQUENCY_2 = 2.0
PERIOD = 2 * pi / FREQUENCY_1
ORBIT_TIMES = linspace(0.0, PERIOD, 721)

DURATION = 10.0
FRAME_RATE = 30

# Where the state arrow is based and how large it is drawn: room for its whole
# orbit in the open space between the two planes
STATE_ORIGIN = (0.35, 0.9, -0.35)
STATE_SCALE = 0.8

# True geometry: the realified qubit, directions (1, 1̇, 2, 2̇) as the geometric indices 1..4
METRIC = euclidean_metric(4)
E_1, F_1, E_2, F_2 = basis_vectors(METRIC)

# Components (η¹, ξ¹, η², ξ²), normalized to ψ^μ ψ^μ = 2
COMPONENTS = array([0.75, 0.95, 0.5, 0.85])
INITIAL_STATE = Vector(sqrt(2) * COMPONENTS / norm(COMPONENTS), grade=1, metric=METRIC)


def evolve(state: Vector, t: float) -> Vector:
    """Phase evolution for time t: rotate each conjugate plane by ω_m t, from e_m toward e_ṁ."""
    R = rotor(E_1 ^ F_1, FREQUENCY_1 * t) * rotor(E_2 ^ F_2, FREQUENCY_2 * t)
    evolved = (R * state * ~R).data[1]

    return evolved


def shadows(state: Vector) -> tuple[Vector, Vector]:
    """Orthogonal projections of the state onto the two conjugate planes: η^m e_m + ξ^m e_ṁ."""
    shadow_1 = state.on[1].data.item() * E_1 + state.on[2].data.item() * F_1
    shadow_2 = state.on[3].data.item() * E_2 + state.on[4].data.item() * F_2

    return shadow_1, shadow_2


@cache
def orbit() -> Vector:
    """The evolved state over one period, as a lot of 4D vectors on a fine time grid (computed once)."""
    times = ORBIT_TIMES
    R = rotor(E_1 ^ F_1, FREQUENCY_1 * times) * rotor(E_2 ^ F_2, FREQUENCY_2 * times)
    states = (R * INITIAL_STATE * ~R).data[1]

    return states


def create_sketch(theme: str = "ink", t: float = 0.0) -> Sketch:
    state = evolve(INITIAL_STATE, t)
    shadow_1, shadow_2 = shadows(state)

    # Depiction: both mode directions along x; plane 1 stands upright, plane 2 lies flat
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
    anchor_1 = array([-2.25, 0.5, 0.75])
    anchor_2 = array([1.9, 0.1, -2.25])
    reach = 1.3

    for anchor, e, f, mode in ((anchor_1, E_1, F_1, "1"), (anchor_2, E_2, F_2, "2")):
        corner = anchor - reach * (sketch.drawn(e) + sketch.drawn(f))
        sketch.plane(e, f, at=corner, span=((0.0, 2 * reach), (0.0, 2 * reach)), grid=6, tone=0.3)
        sketch.vector(0.5 * e, at=corner + 0.75 * sketch.drawn(e), weight="fine", head=0.8)
        sketch.vector(0.5 * f, at=corner + 0.75 * sketch.drawn(f), weight="fine", head=0.8)
        sketch.label(r"$\mathbf{e}_{" + mode + "}$", corner + 1.0 * sketch.drawn(e), offset=(3, -13), size=14)
        sketch.label(
            r"$\mathbf{e}_{\overset{\bullet}{" + mode + "}}$", corner + 1.0 * sketch.drawn(f), offset=(-17, 3), size=14
        )

    # Phase circles through each shadow, arrowed in the right-handed sense of the evolution
    sketch.circle(anchor_1, E_1, F_1, radius=norm(shadow_1.data), arc=(1.0, 2.6 + pi), arrow=True)
    sketch.circle(anchor_2, E_2, F_2, radius=norm(shadow_2.data), arc=(1.25, 2.85 + pi), arrow=True)

    # The drawn state's closed orbit over one period, and the part already traversed
    path = origin + scale * sketch.drawn(orbit())
    sketch.curve(path, dashed=True, weight="hair", level=0.35)
    elapsed = searchsorted(ORBIT_TIMES, t % PERIOD)
    traversed = concatenate([path[:elapsed], [origin + scale * sketch.drawn(state)]])
    if len(traversed) > 1:
        sketch.curve(traversed, level=0.5)

    tip_1 = sketch.vector(
        shadow_1, at=anchor_1, weight="bold", label=r"$\mathbf{ψ}_1$", label_offset=-13, label_size=14
    )
    tip_2 = sketch.vector(
        shadow_2, at=anchor_2, weight="bold", label=r"$\mathbf{ψ}_2$", label_offset=-17, label_size=14
    )
    sketch.point(anchor_1)
    sketch.point(anchor_2)

    tip = sketch.vector(
        state, at=origin, scale=scale, weight="heavy", label=r"$\mathbf{ψ}$", label_offset=16, label_size=14
    )
    sketch.point(origin, radius=4.0)

    for start, end in ((origin, anchor_1), (origin, anchor_2), (tip, tip_1), (tip, tip_2)):
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
        path = animate(lambda t: create_sketch(theme, t), times, output, fps=FRAME_RATE)
    else:
        path = create_sketch(theme).save(output)

    print(f"Saved {path}")
