"""
Trivector Rotation Animation

Builds up a trivector from basis elements, then rotates it:
  e1 -> e2 -> e12 -> e3 -> e123 -> rotate 4*pi

The script controls all geometry. The animation just observes and records.

Run: uv run python -m morphis.examples.animate_trivector_rotation
"""

from numpy import array, pi, sqrt, zeros

from morphis.ga.constructors import basis_vectors
from morphis.ga.model import Blade
from morphis.ga.motors import Motor
from morphis.visualization.animation import Animation


RED = (0.85, 0.2, 0.2)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out: slow start, fast middle, slow end."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def main():
    # Build basis
    e1, e2, e3 = basis_vectors(dim=3)
    e12 = e1 ^ e2
    e123 = e1 ^ e2 ^ e3

    # Timeline (in seconds)
    # Phase 1: Build up (50% longer fade times)
    fade_dur = 0.45  # was 0.3, now 50% longer

    t_e1_in = 0.0
    t_e2_in = 1.0
    t_e12_in = 2.0
    t_e1_out = 2.0   # e1 and e2 fade out when e12 appears
    t_e2_out = 2.0
    t_e3_in = 3.0
    t_e123_in = 4.0
    t_e12_out = 4.0  # e12 and e3 fade out when e123 appears
    t_e3_out = 4.0

    # Phase 2: Rotate
    t_rotate_start = 5.0
    t_rotate_end = 8.0

    total_duration = t_rotate_end
    fps = 60

    # Rotation setup
    s = 1.0 / sqrt(3.0)
    diagonal = s * (e12 + (e2 ^ e3) + (e3 ^ e1))

    # Embed bivector to PGA
    dim_pga = 4
    B_pga_data = zeros((dim_pga, dim_pga))
    for i in range(3):
        for j in range(3):
            B_pga_data[i + 1, j + 1] = diagonal.data[i, j]
    B_pga = Blade(data=B_pga_data, grade=2, dim=dim_pga, cdim=0)

    total_rotation = 4 * pi
    rotation_duration = t_rotate_end - t_rotate_start

    # Create animation
    anim = Animation(fps=fps, theme="obsidian", size=(800, 600))

    # Track all objects
    anim.track(e1, color=RED)
    anim.track(e2, color=RED)
    anim.track(e12, color=RED)
    anim.track(e3, color=RED)
    anim.track(e123, color=RED)

    # Focus on center of cube, camera offset from diagonal for visual interest
    focal = 1.0 / sqrt(3.0)
    anim.camera(position=(4.2, -2.4, 3.5), focal_point=(focal, focal, focal))

    # Schedule effects for build-up sequence
    anim.fade_in(e1, t=t_e1_in, duration=fade_dur)
    anim.fade_out(e1, t=t_e1_out, duration=fade_dur)

    anim.fade_in(e2, t=t_e2_in, duration=fade_dur)
    anim.fade_out(e2, t=t_e2_out, duration=fade_dur)

    anim.fade_in(e12, t=t_e12_in, duration=fade_dur)
    anim.fade_out(e12, t=t_e12_out, duration=fade_dur)

    anim.fade_in(e3, t=t_e3_in, duration=fade_dur)
    anim.fade_out(e3, t=t_e3_out, duration=fade_dur)

    anim.fade_in(e123, t=t_e123_in, duration=fade_dur)

    print("Trivector Rotation Animation")
    print("=" * 40)
    print("Phase 1: Build up")
    print("  [0-1s] e1")
    print("  [1-2s] e2")
    print("  [2-3s] e12 = e1 ^ e2")
    print("  [3-4s] e3")
    print("  [4-5s] e123 = e1 ^ e2 ^ e3")
    print("Phase 2: Rotate")
    print("  [5-8s] Rotate 4*pi (cubic easing)")
    print()

    # Initial vectors for e123 (will be transformed during rotation)
    initial_vectors = array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Animation loop
    anim.start()
    n_frames = int(total_duration * fps)

    for frame in range(n_frames + 1):
        t = frame / fps

        # During rotation phase, apply smoothed rotation
        if t >= t_rotate_start:
            # Compute eased progress
            progress = (t - t_rotate_start) / rotation_duration
            progress = min(progress, 1.0)
            eased = ease_in_out_cubic(progress)

            # Compute total angle at this point
            current_angle = total_rotation * eased

            # Apply rotation from initial state
            M = Motor.rotor(B_pga, current_angle)
            vectors = M.apply_to_euclidean(initial_vectors)

            # Tell animation the current vectors (bypasses factor_blade)
            anim.set_vectors(e123, vectors)

        anim.capture(t)

    print("Playing animation...")
    anim.play()


def save_demo():
    """Generate the demo GIF for the README."""
    e1, e2, e3 = basis_vectors(dim=3)
    e12 = e1 ^ e2
    e123 = e1 ^ e2 ^ e3

    # Faster timing for GIF (but still 50% longer fades)
    fps = 30
    fade_dur = 0.3  # was 0.2, now 50% longer

    t_e1_in = 0.0
    t_e2_in = 0.6
    t_e12_in = 1.2
    t_e1_out = 1.2   # e1 and e2 fade out when e12 appears
    t_e2_out = 1.2
    t_e3_in = 1.8
    t_e123_in = 2.4
    t_e12_out = 2.4  # e12 and e3 fade out when e123 appears
    t_e3_out = 2.4
    t_rotate_start = 3.0
    t_rotate_end = 5.0
    t_pause_end = 8.0  # Hold final state for 3 seconds

    total_duration = t_pause_end

    s = 1.0 / sqrt(3.0)
    diagonal = s * ((e1 ^ e2) + (e2 ^ e3) + (e3 ^ e1))

    dim_pga = 4
    B_pga_data = zeros((dim_pga, dim_pga))
    for i in range(3):
        for j in range(3):
            B_pga_data[i + 1, j + 1] = diagonal.data[i, j]
    B_pga = Blade(data=B_pga_data, grade=2, dim=dim_pga, cdim=0)

    total_rotation = 4 * pi
    rotation_duration = t_rotate_end - t_rotate_start

    anim = Animation(fps=fps, theme="obsidian", size=(400, 300))

    anim.track(e1, color=RED)
    anim.track(e2, color=RED)
    anim.track(e12, color=RED)
    anim.track(e3, color=RED)
    anim.track(e123, color=RED)

    # Focus on center of cube, camera offset from diagonal for visual interest
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

    initial_vectors = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    anim.start()
    n_frames = int(total_duration * fps)

    for frame in range(n_frames + 1):
        t = frame / fps

        if t >= t_rotate_start:
            progress = min((t - t_rotate_start) / rotation_duration, 1.0)
            eased = ease_in_out_cubic(progress)
            current_angle = total_rotation * eased

            M = Motor.rotor(B_pga, current_angle)
            vectors = M.apply_to_euclidean(initial_vectors)

            anim.set_vectors(e123, vectors)

        anim.capture(t)

    anim.save("figures/rotating_trivector.gif", loop=True)
    print("Saved figures/rotating_trivector.gif")


if __name__ == "__main__":
    import sys

    if "--save" in sys.argv:
        save_demo()
    else:
        main()
