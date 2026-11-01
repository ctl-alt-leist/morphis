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

The Scene draws the tesseract in perspective with the eye on the e_4 axis at
distance D, a point x drawn at D x / (D - x_4). The projective parameter is the
inverse distance κ = 1 / D, the weight row of the map: at κ = 0 the eye is at
infinity and the map is the orthographic one onto e_1, e_2, e_3, where the cell
is a cube whose far and near halves overlap. Sliding κ up brings the eye in
from infinity, and the far cell shrinks continuously inside the near one. Near
κ = 0 the picture moves linearly in κ, so easing κ itself gives an even slide.

The tesseract makes a quarter turn orthographically, coming to rest square to
the axes, where the two cubes at x_4 = ±½ are drawn on top of each other. It
holds still while κ slides up, so the change in the picture is the eye coming
in alone, and the far cube shrinks inside the near one. Then it makes a full
turn in perspective, where the turn carries cells through each other.
A caption in the corner names the view and gives κ. Each corner is
mapped by the projective map itself, so the drawn cells stay exact.

Run: uv run python -m morphis.examples.tesseract
Record: uv run python -m morphis.examples.tesseract figures/tesseract/tesseract.gif
"""

import sys
from contextlib import nullcontext

from numpy import pi

from morphis.elements import Frame, basis_vectors, euclidean_metric
from morphis.transforms import rotor
from morphis.utils.easing import ease_in_out_cubic
from morphis.visuals import RED, Projective, Scene


FRAME_RATE = 30
DURATION_FADE_IN = 1.0
DURATION_QUARTER_TURN = 4.0
DURATION_SLIDE = 3.0
DURATION_FULL_TURN = 6.0

EYE_DISTANCE = 3.0
CAMERA_POSITION = (4.2, -3.6, 2.8)


def view(κ: float) -> Projective:
    """Perspective from the eye at distance 1 / κ along e_4; κ = 0 is the orthographic view."""
    return Projective(eye=κ)


def caption(κ: float) -> str:
    """The view and its projective parameter: orthographic at κ = 0, perspective once κ is drawn above it."""
    value = f"{κ:.2f}"
    name = "orthographic" if float(value) == 0 else "perspective"

    return f"{name}, κ = {value}"


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

    scene = Scene(frame_rate=FRAME_RATE, window=output is None, projection=view(0.0))
    scene.add(edges, color=RED, filled=True, origin=corner)
    scene.fade_in(edges, t=0.0, duration=DURATION_FADE_IN)
    scene.caption(caption(0.0))
    scene.camera(position=CAMERA_POSITION, focal_point=(0.0, 0.0, 0.0))

    def turn(angle: float) -> None:
        R = rotor(B, angle)
        edges.data[...] = start_edges.transform(R).data
        corner.data[...] = (R * start_corner * ~R)[1].data

    recording = scene.record(output) if output else nullcontext()
    with recording:
        t = 0.0
        dt = 1.0 / FRAME_RATE
        for _ in range(int(DURATION_FADE_IN * FRAME_RATE)):
            scene.capture(t)
            t += dt

        # (start angle, end angle, start κ, end κ, duration): a quarter turn orthographically, the eye coming
        # in with the cell at rest, then a full turn in perspective; each step is eased, so the cell rests between
        eye = 1.0 / EYE_DISTANCE
        steps = (
            (0.0, pi / 2, 0.0, 0.0, DURATION_QUARTER_TURN),
            (pi / 2, pi / 2, 0.0, eye, DURATION_SLIDE),
            (pi / 2, 5 * pi / 2, eye, eye, DURATION_FULL_TURN),
        )
        for angle_start, angle_end, κ_start, κ_end, duration in steps:
            n_frames = int(duration * FRAME_RATE)
            for n in range(n_frames + 1):
                progress = ease_in_out_cubic(n / n_frames)
                κ = κ_start + (κ_end - κ_start) * progress
                scene.set_projection(view(κ))
                scene.caption(caption(κ))
                turn(angle_start + (angle_end - angle_start) * progress)
                scene.capture(t)
                t += dt

    return scene


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    scene = create_scene(output)
    if output:
        print(f"Saved {output}")
    scene.show()
