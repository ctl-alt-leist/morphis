"""
Conjugate Planes

A qubit state realified: ℂ² becomes ℝ⁴ with directions 1, 1̇, 2, 2̇, paired
into two conjugate planes, one per mode. The state ψ splits exactly into its
shadows ψ_a and ψ_b in the two planes. The figure then chooses how to draw
this 4D arrangement in three dimensions: both representative directions are
drawn along the same page direction, and each plane is set out at its own
place in the space, joined to the state by dashed construction lines.

Run: uv run python -m morphis.examples.conjugate_planes [output.png] [ink|parchment|chalkboard]
"""

import sys

from numpy import array, pi
from numpy.linalg import norm

from morphis.elements import Vector, basis_vectors, euclidean_metric
from morphis.visuals.ink import Camera, Depiction, OrganicSpace, Sketch


def create_sketch(theme: str = "ink") -> Sketch:
    # True geometry: the realified qubit, directions (1, 1̇, 2, 2̇) as e_1..e_4
    g = euclidean_metric(4)
    f_a, g_a, f_b, g_b = basis_vectors(g)

    state = Vector([0.75, 0.95, 0.5, 0.85], grade=1, metric=g)
    shadow_a = state.on[1].data.item() * f_a + state.on[2].data.item() * g_a
    shadow_b = state.on[3].data.item() * f_b + state.on[4].data.item() * g_b

    # Depiction: both representative directions along x; plane a stands upright, plane b lies flat
    depiction = Depiction(g, {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})

    sketch = Sketch(
        camera=Camera(azimuth=-24.0, elevation=34.0, distance=16.0, fov=16.0),
        depiction=depiction,
        size=(8.0, 6.0),
        seed=4,
        theme=theme,
    )
    sketch.space(OrganicSpace(seed=3, stretch=(1.6, 1.25, 1.05), size=3.2))

    # Placement: where each piece sits in the drawing space
    origin = array([-0.95, -0.9, -1.1])
    anchor_a = array([-2.55, 0.5, 0.45])
    anchor_b = array([2.2, 0.1, -2.0])
    reach = 1.3

    for anchor, u, v, mode in ((anchor_a, f_a, g_a, "a"), (anchor_b, f_b, g_b, "b")):
        corner = anchor - reach * (sketch.drawn(u) + sketch.drawn(v))
        sketch.plane(u, v, at=corner, span=((0.0, 2 * reach), (0.0, 2 * reach)), grid=6, tone=0.3)
        sketch.vector(0.5 * u, at=corner + 0.75 * sketch.drawn(u), weight="fine", head=0.8)
        sketch.vector(0.5 * v, at=corner + 0.75 * sketch.drawn(v), weight="fine", head=0.8)
        sketch.label(rf"$f_{mode}$", corner + 1.0 * sketch.drawn(u), offset=(3, -13))
        sketch.label(rf"$g_{mode}$", corner + 1.0 * sketch.drawn(v), offset=(-12, 3))

    tip_a = sketch.vector(shadow_a, at=anchor_a, weight="bold")
    tip_b = sketch.vector(shadow_b, at=anchor_b, weight="bold")
    sketch.circle(anchor_a, f_a, g_a, radius=norm(shadow_a.data), arc=(1.0, 2.6 + pi), arrow=True)
    sketch.circle(anchor_b, f_b, g_b, radius=norm(shadow_b.data), arc=(1.25, 2.85 + pi), arrow=True)
    sketch.point(anchor_a)
    sketch.point(anchor_b)

    tip = sketch.vector(state, at=origin, scale=1.55, weight="heavy")
    sketch.point(origin, radius=4.0)
    sketch.point(tip, radius=2.4)
    sketch.point(tip_a, radius=2.2)
    sketch.point(tip_b, radius=2.2)

    for start, end in ((origin, anchor_a), (origin, anchor_b), (tip, tip_a), (tip, tip_b)):
        sketch.line(start, end)

    sketch.label("$ψ$", tip, offset=(15, 4), size=21)
    sketch.label("$ψ_a$", tip_a, offset=(17, -6), size=17)
    sketch.label("$ψ_b$", tip_b, offset=(17, 4), size=17)
    sketch.label("conjugate plane $a$", anchor_a + array([0.0, 0.0, reach]), offset=(-10, 26), size=14)
    sketch.label("conjugate plane $b$", anchor_b + array([reach, -reach, 0.0]), offset=(40, -34), size=14)
    sketch.label("state space", array([0.8, 0.0, 2.55]), offset=(0, 0), size=15)

    return sketch


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "conjugate-planes.png"
    theme = sys.argv[2] if len(sys.argv) > 2 else "ink"
    path = create_sketch(theme).save(output)
    print(f"Saved {path}")
