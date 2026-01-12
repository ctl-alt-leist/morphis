"""
Trivector Rotation Animation

Builds up a trivector from basis elements, then rotates it:
  e1 -> e2 -> e12 -> e3 -> e123 -> rotate 4*pi

Run: uv run python -m morphis.examples.animate_trivector_rotation
"""

from numpy import pi, sqrt, zeros

from morphis.ga.constructors import basis_vectors
from morphis.ga.model import Blade
from morphis.ga.motors import Motor
from morphis.geometry.projective import euclidean as to_euclidean, point
from morphis.utils.easing import ease_in_out_cubic
from morphis.visualization.animation import Animation
from morphis.visualization.theme import RED


# Timeline constants (seconds)
FADE_DUR = 0.45
T_E1_IN, T_E1_OUT = 0.0, 2.0
T_E2_IN, T_E2_OUT = 1.0, 2.0
T_E12_IN, T_E12_OUT = 2.0, 4.0
T_E3_IN, T_E3_OUT = 3.0, 4.0
T_E123_IN = 4.0
T_ROTATE_START, T_ROTATE_END = 5.0, 8.0
TOTAL_ROTATION = 4 * pi
FPS = 60


def embed_bivector_to_pga(B: Blade) -> Blade:
    """Embed a 3D Euclidean bivector into 4D PGA."""
    dim_pga = B.dim + 1
    B_pga_data = zeros((dim_pga, dim_pga))
    B_pga_data[1:, 1:] = B.data
    return Blade(data=B_pga_data, grade=2, dim=dim_pga, cdim=0)


def apply_motor_to_euclidean(motor: Motor, vectors):
    """Apply motor to Euclidean vectors via PGA embedding."""
    results = []
    for v in vectors:
        p = point(v)
        p_transformed = motor.apply(p)
        results.append(to_euclidean(p_transformed))
    return results


def main():
    # Build basis
    e1, e2, e3 = basis_vectors(dim=3)
    e12 = e1 ^ e2
    e123 = e1 ^ e2 ^ e3

    # Rotation setup: diagonal bivector for rotation plane
    s = 1.0 / sqrt(3.0)
    diagonal = s * (e12 + (e2 ^ e3) + (e3 ^ e1))
    B_pga = embed_bivector_to_pga(diagonal)

    # Initial vectors for e123 visualization
    initial_vectors = [e1.data, e2.data, e3.data]

    # Create animation
    anim = Animation(fps=FPS, theme="obsidian", size=(1200, 900))

    # Track all objects
    anim.track(e1, e2, e12, e3, e123, color=RED)

    # Camera position
    focal = 1.0 / sqrt(3.0)
    anim.camera(position=(4.2, -2.4, 3.5), focal_point=(focal, focal, focal))

    # Schedule effects for build-up sequence
    anim.fade_in(e1, t=T_E1_IN, duration=FADE_DUR)
    anim.fade_out(e1, t=T_E1_OUT, duration=FADE_DUR)
    anim.fade_in(e2, t=T_E2_IN, duration=FADE_DUR)
    anim.fade_out(e2, t=T_E2_OUT, duration=FADE_DUR)
    anim.fade_in(e12, t=T_E12_IN, duration=FADE_DUR)
    anim.fade_out(e12, t=T_E12_OUT, duration=FADE_DUR)
    anim.fade_in(e3, t=T_E3_IN, duration=FADE_DUR)
    anim.fade_out(e3, t=T_E3_OUT, duration=FADE_DUR)
    anim.fade_in(e123, t=T_E123_IN, duration=FADE_DUR)

    print("Trivector Rotation Animation")
    print("=" * 40)
    print("Phase 1: Build up e1 -> e2 -> e12 -> e3 -> e123")
    print("Phase 2: Rotate 4π around diagonal")
    print()

    # Animation loop
    anim.start()
    n_frames = int(T_ROTATE_END * FPS)
    rotation_duration = T_ROTATE_END - T_ROTATE_START

    for frame in range(n_frames + 1):
        t = frame / FPS

        # During rotation phase, apply smoothed rotation
        if t >= T_ROTATE_START:
            progress = min((t - T_ROTATE_START) / rotation_duration, 1.0)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)

            # Apply motor to get rotated vectors
            M = Motor.rotor(B_pga, angle)
            rotated = apply_motor_to_euclidean(M, initial_vectors)
            anim.set_vectors(e123, rotated)

        anim.capture(t)

    print("Playing animation...")
    anim.play()


def save_demo():
    """Generate the demo GIF for the README."""
    e1, e2, e3 = basis_vectors(dim=3)
    e12 = e1 ^ e2
    e123 = e1 ^ e2 ^ e3

    # Faster timing for GIF
    fps = 30
    fade_dur = 0.3
    t_e1_in, t_e1_out = 0.0, 1.2
    t_e2_in, t_e2_out = 0.6, 1.2
    t_e12_in, t_e12_out = 1.2, 2.4
    t_e3_in, t_e3_out = 1.8, 2.4
    t_e123_in = 2.4
    t_rotate_start, t_rotate_end = 3.0, 5.0
    t_pause_end = 8.0

    s = 1.0 / sqrt(3.0)
    diagonal = s * ((e1 ^ e2) + (e2 ^ e3) + (e3 ^ e1))
    B_pga = embed_bivector_to_pga(diagonal)

    initial_vectors = [e1.data, e2.data, e3.data]
    rotation_duration = t_rotate_end - t_rotate_start

    anim = Animation(fps=fps, theme="obsidian", size=(400, 300))
    anim.track(e1, e2, e12, e3, e123, color=RED)

    focal = 1.0 / sqrt(3.0)
    anim.camera(position=(4.2, -2.4, 3.5), focal_point=(focal, focal, focal))

    anim.fade_in(e1, t=t_e1_in, duration=fade_dur)
    anim.fade_out(e1, t=t_e1_out, duration=fade_dur)
    anim.fade_in(e2, t=t_e2_in, duration=fade_dur)
    anim.fade_out(e2, t=t_e2_out, duration=fade_dur)
    anim.fade_in(e12, t=t_e12_in, duration=fade_dur)
    anim.fade_out(e12, t=t_e12_out, duration=fade_dur)
    anim.fade_in(e3, t=t_e3_in, duration=fade_dur)
    anim.fade_out(e3, t=t_e3_out, duration=fade_dur)
    anim.fade_in(e123, t=t_e123_in, duration=fade_dur)

    anim.start()
    n_frames = int(t_pause_end * fps)

    for frame in range(n_frames + 1):
        t = frame / fps

        if t >= t_rotate_start:
            progress = min((t - t_rotate_start) / rotation_duration, 1.0)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B_pga, angle)
            rotated = apply_motor_to_euclidean(M, initial_vectors)
            anim.set_vectors(e123, rotated)

        anim.capture(t)

    anim.save("figures/rotating_trivector.gif", loop=True)
    print("Saved figures/rotating_trivector.gif")


if __name__ == "__main__":
    import sys

    if "--save" in sys.argv:
        save_demo()
    else:
        main()
