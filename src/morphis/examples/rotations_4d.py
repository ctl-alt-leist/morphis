"""
4D Rotation Animation

Visualizes a 4D frame rotating with projection to 3D:
- Selectable projection axes (e123, e234, etc.)
- Rotation in arbitrary bivector planes

Run: uv run python -m morphis.examples.rotations_4d
Record: uv run python -m morphis.examples.rotations_4d figures/rotations-4d/rotations-4d.gif
"""

import sys
from contextlib import nullcontext

from numpy import array, diff, pi, sqrt
from numpy.linalg import norm

from morphis.elements import Frame, basis_vectors, euclidean_metric
from morphis.operations import unit
from morphis.transforms import rotor
from morphis.utils.easing import ease_in_out_cubic
from morphis.visuals import RED, Scene


# Configuration
FRAME_RATE = 60
DURATION_FADE_IN = 1.0
DURATION_ROTATE = 2.0
TOTAL_ROTATION = 2 * pi


def compute_delta_angles(duration: float, total_angle: float) -> list[float]:
    """Pre-compute incremental angles for eased rotation."""
    n = int(duration * FRAME_RATE)
    angles = [total_angle * ease_in_out_cubic((i + 1) / n) for i in range(n)]
    return list(diff([0.0] + angles))


def create_scene(output: str | None = None) -> Scene:
    """
    Create the 4D frame rotation animation.

    Shown live in a window, or, given an output path (.gif or .mp4), recorded
    off screen to that file.
    """
    # Create 4D basis
    g = euclidean_metric(4)
    e1, e2, e3, e4 = basis_vectors(g)

    # Rotation bivectors
    b1 = unit((e1 ^ e3) + (e2 ^ e3))
    b2 = unit((e1 ^ e4) + (e2 ^ e4) + (e3 ^ e4))

    # The ONE frame we animate
    F = Frame(e1, e2, e3, e4)

    # Pre-compute delta angles for eased rotation
    d_angles = compute_delta_angles(DURATION_ROTATE, TOTAL_ROTATION)

    # Create scene
    scene = Scene(
        frame_rate=FRAME_RATE,
        theme="obsidian",
        window=output is None,
        projection=(1, 2, 3),
    )
    scene.add(F, color=RED, filled=True)
    scene.fade_in(F, t=0.0, duration=DURATION_FADE_IN)

    # Camera setup
    focal_point = array([1.0, 1.0, 1.0]) * 0.0 / sqrt(2)
    direction = array([+0.9, -0.8, 0.7])
    distance = 7.0
    position = tuple(array(focal_point) + distance * direction / norm(direction))
    scene.camera(position=position, focal_point=focal_point)

    print("4D Frame Rotation Animation")
    print("=" * 40)
    print(f"Fade in: {DURATION_FADE_IN}s")
    print(f"4 rotation phases: {DURATION_ROTATE}s each")
    print()

    recording = scene.record(output) if output else nullcontext()
    with recording:
        t = 0.0
        dt = 1.0 / FRAME_RATE

        # Fade in
        for _ in range(int(DURATION_FADE_IN * FRAME_RATE) + 1):
            scene.capture(t)
            t += dt

        # First two rotations in e123 projection
        for b in [b1, b2]:
            for d_angle in d_angles:
                M = rotor(b, d_angle)
                F.data[...] = F.transform(M).data
                scene.capture(t)
                t += dt

        # Switch to e234 projection
        scene.set_projection((2, 3, 4))

        # Last two rotations in e234 projection
        for b in [b1, b2]:
            for d_angle in d_angles:
                M = rotor(b, d_angle)
                F.data[...] = F.transform(M).data
                scene.capture(t)
                t += dt

    return scene


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    scene = create_scene(output)
    if output:
        print(f"Saved {output}")
    scene.show()
