"""
Tesseract

The unit tesseract of ℝ⁴, centered on the origin, turning in an isoclinic
double rotation: the bivector

    B = e_14 + e_23

is not simple. It is the sum of two rotations in the orthogonal planes e_14 and
e_23 at the same rate, so every vector turns by the same angle θ, and its rotor

    R(θ) = exp(-B θ / 2)

has a grade-4 part. After θ = 2π both planes have made a full turn, and the
rotor is 1 again. exp_vector evaluates it in closed form through the
invariants of B. The tesseract's corner and its four edge vectors are turned by
the sandwich R x R̃ each frame; the Scene draws what it is handed.

The Scene draws the tesseract first orthographically onto e_1, e_2, e_3, where
the cell is a cube whose far and near halves overlap, and then in perspective
with the eye on the e_4 axis, where the far cell is drawn inside the near one
and the turn carries cells through each other. The perspective turn picks up
where the orthographic quarter turn left off, so the jump at the switch is the
change of view alone, and a caption in the corner names the view in use. Under
perspective each corner is mapped by the projective map itself, so the drawn
cells stay exact.

Run: uv run python -m morphis.examples.tesseract
Record: uv run python -m morphis.examples.tesseract figures/tesseract/tesseract.gif
"""

import sys
from contextlib import nullcontext

from numpy import pi

from morphis.elements import Frame, basis_vectors, euclidean_metric
from morphis.transforms import rotor
from morphis.utils.easing import ease_in_out_cubic
from morphis.visuals import RED, Orthographic, Perspective, Scene


FRAME_RATE = 30
DURATION_FADE_IN = 1.0
DURATION_TURN = 4.0

EYE_DISTANCE = 3.0
CAMERA_POSITION = (4.2, -3.6, 2.8)


def create_scene(output: str | None = None) -> Scene:
    """
    Create the tesseract animation.

    Shown live in a window, or, given an output path (.gif or .mp4), recorded
    off screen to that file.
    """
    g = euclidean_metric(4)
    e1, e2, e3, e4 = basis_vectors(g)
    B = (e1 ^ e4) + (e2 ^ e3)

    # The cell spanned by the four edge vectors from one corner, centered on the origin
    edges = Frame(e1, e2, e3, e4)
    corner = -0.5 * (e1 + e2 + e3 + e4)
    start_edges, start_corner = edges.copy(), corner.copy()

    scene = Scene(frame_rate=FRAME_RATE, window=output is None, projection=Orthographic(axes=(1, 2, 3)))
    scene.add(edges, color=RED, filled=True, origin=corner)
    scene.fade_in(edges, t=0.0, duration=DURATION_FADE_IN)
    scene.caption("projection: orthographic")
    scene.camera(position=CAMERA_POSITION, focal_point=(0.0, 0.0, 0.0))

    def turn(angle: float) -> None:
        R = rotor(B, angle)
        edges.data[...] = start_edges.transform(R).data
        corner.data[...] = (R * start_corner * ~R)[1].data

    recording = scene.record(output) if output else nullcontext()
    with recording:
        t = 0.0
        dt = 1.0 / FRAME_RATE
        n_turn = int(DURATION_TURN * FRAME_RATE)

        for _ in range(int(DURATION_FADE_IN * FRAME_RATE)):
            scene.capture(t)
            t += dt

        # A quarter turn drawn orthographically, then a full turn more in perspective
        phases = (
            ("orthographic", Orthographic(axes=(1, 2, 3)), 0.0, pi / 2),
            ("perspective", Perspective(distance=EYE_DISTANCE), pi / 2, 2 * pi),
        )
        for name, view, start, total in phases:
            scene.set_projection(view)
            scene.caption(f"projection: {name}")
            for n in range(n_turn + 1):
                turn(start + total * ease_in_out_cubic(n / n_turn))
                scene.capture(t)
                t += dt

    return scene


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    scene = create_scene(output)
    if output:
        print(f"Saved {output}")
    scene.show()
