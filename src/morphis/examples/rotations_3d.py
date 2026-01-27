"""
3D Rotation Animation

Visualizes a 3D frame rotating around a diagonal axis.
The frame is displayed as three arrows from the origin.

Run: uv run python -m morphis.examples.rotations_3d
"""

import argparse

from numpy import diff, pi

from morphis.elements import Frame, basis_vectors, euclidean_metric
from morphis.operations import normalize
from morphis.transforms import rotor
from morphis.utils.easing import ease_in_out_cubic
from morphis.visuals import RED, Animation


# Configuration
FRAME_RATE = 60
DURATION_FADE_IN = 0.5
DURATION_ROTATE = 4.0
TOTAL_ROTATION = 4 * pi


def compute_delta_angles(duration: float, total_angle: float) -> list[float]:
    """Pre-compute incremental angles for eased rotation."""
    n = int(duration * FRAME_RATE)
    angles = [total_angle * ease_in_out_cubic((i + 1) / n) for i in range(n)]
    return list(diff([0.0] + angles))


def create_animation():
    """
    Create the 3D frame rotation animation.

    Returns configured Animation ready for play() or save().
    """
    # Build basis vectors
    g = euclidean_metric(3)
    e1, e2, e3 = basis_vectors(g)

    # Rotation bivector: diagonal plane
    b = normalize((e1 ^ e2) + (e2 ^ e3) + (e3 ^ e1))

    # The ONE frame we animate
    F = Frame(e1, e2, e3)

    # Pre-compute delta angles for eased rotation
    d_angles = compute_delta_angles(DURATION_ROTATE, TOTAL_ROTATION)

    # Create animation
    anim = Animation(
        frame_rate=FRAME_RATE,
        theme="obsidian",
        size=(1200, 900),
        auto_camera=True,
    )
    anim.watch(F, color=RED, filled=True)
    anim.fade_in(F, t=0.0, duration=DURATION_FADE_IN)

    print("3D Frame Rotation Animation")
    print("=" * 40)
    print(f"Fade in: {DURATION_FADE_IN}s")
    print(f"Rotate 4π: {DURATION_ROTATE}s")
    print()

    anim.start()
    t = 0.0
    dt = 1.0 / FRAME_RATE

    # Fade in
    for _ in range(int(DURATION_FADE_IN * FRAME_RATE) + 1):
        anim.capture(t)
        t += dt

    # Rotation
    for d_angle in d_angles:
        M = rotor(b, d_angle)
        F.data[...] = F.transform(M).data
        anim.capture(t)
        t += dt

    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D frame rotation animation")
    parser.add_argument("--save", type=str, help="Also save to file (e.g., out.gif)")
    args = parser.parse_args()

    anim = create_animation()

    print("Playing animation... (close window when done)")
    anim.play()

    if args.save:
        anim.save(args.save)
        print(f"Saved {args.save}")
