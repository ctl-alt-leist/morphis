"""
4D Blade Rotation Animation

Demonstrates visualization of a 4-blade (quadvector) in 4D space with:
- Orthogonal projection to 3D with selectable axes (e123, e234, etc.)
- Rotation in arbitrary bivector planes through the object center
- View switching during animation

Run: uv run python -m morphis.examples.animate_4d_rotation
"""

import argparse
from math import sqrt

from numpy import pi, zeros

from morphis.ga.constructors import basis_vectors
from morphis.ga.model import Blade
from morphis.ga.motors import Motor
from morphis.geometry.projective import euclidean as to_euclidean, point
from morphis.utils.easing import ease_in_out_cubic
from morphis.visualization.animation import Animation
from morphis.visualization.theme import RED


# Timeline constants (seconds)
FPS = 60
T_FADE_END = 0.5
T_ROT1_END = 1.5
T_ROT2_END = 2.5
T_ROT3_END = 3.5
T_ROT4_END = 4.5
TOTAL_ROTATION = 2 * pi


def embed_bivector_to_pga(B: Blade) -> Blade:
    """Embed a Euclidean bivector into PGA (add ideal dimension at index 0)."""
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


def create_rotation_bivector(direction_indices: list[int], target_index: int, dim: int) -> Blade:
    """
    Create a normalized bivector for rotation in (sum of directions) ^ target plane.

    Args:
        direction_indices: Indices of vectors to sum (e.g., [0, 1] for e1+e2)
        target_index: Index of target direction (e.g., 2 for e3)
        dim: Dimension of space

    Returns:
        Normalized PGA bivector
    """
    dim_pga = dim + 1
    B_data = zeros((dim_pga, dim_pga))

    # Normalization factor
    s = 1.0 / sqrt(len(direction_indices))

    # Create wedge product (sum of directions) ^ target
    for i in direction_indices:
        # PGA indices are offset by 1 (index 0 is ideal)
        pga_i = i + 1
        pga_t = target_index + 1
        B_data[pga_i, pga_t] = s
        B_data[pga_t, pga_i] = -s

    return Blade(data=B_data, grade=2, dim=dim_pga, cdim=0)


def main():
    """Run the 4D rotation animation interactively."""
    # Create 4D basis and quadvector
    e1, e2, e3, e4 = basis_vectors(dim=4)
    Q = e1 ^ e2 ^ e3 ^ e4

    # Initial spanning vectors
    initial_vectors = [e1.data, e2.data, e3.data, e4.data]
    origin = zeros(4)

    # Rotation bivectors: (e1+e2)^e3 and (e1+e2+e3)^e4
    B1 = create_rotation_bivector([0, 1], 2, dim=4)  # (e1+e2)^e3
    B2 = create_rotation_bivector([0, 1, 2], 3, dim=4)  # (e1+e2+e3)^e4

    # Create animation
    anim = Animation(fps=FPS, theme="obsidian", size=(1200, 900), show_basis=True)
    anim.track(Q, color=RED)
    anim.set_projection(Q, (0, 1, 2))
    anim.set_vectors(Q, initial_vectors, origin)
    anim.set_basis_labels((r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_3$"))
    anim.fade_in(Q, t=0.0, duration=T_FADE_END)
    anim.camera(position=(4.62, -2.64, 3.85), focal_point=(0, 0, 0))

    print("4D Blade Rotation Animation")
    print("=" * 40)
    print(f"[0-{T_FADE_END}s] Fade in (e123 projection)")
    print(f"[{T_FADE_END}-{T_ROT1_END}s] Rotate in (e1+e2)^e3 plane")
    print(f"[{T_ROT1_END}-{T_ROT2_END}s] Rotate in (e1+e2+e3)^e4 plane")
    print(f"[{T_ROT2_END}s] Switch to e234 projection")
    print(f"[{T_ROT2_END}-{T_ROT3_END}s] Rotate in (e1+e2)^e3 plane")
    print(f"[{T_ROT3_END}-{T_ROT4_END}s] Rotate in (e1+e2+e3)^e4 plane")
    print()

    anim.start()
    n_frames = int(T_ROT4_END * FPS)
    current_projection = (0, 1, 2)

    # Precompute full rotation motors
    M1_full = Motor.rotor(B1, TOTAL_ROTATION)
    M2_full = Motor.rotor(B2, TOTAL_ROTATION)

    for frame in range(n_frames + 1):
        t = frame / FPS

        if t < T_FADE_END:
            vectors = initial_vectors
        elif t < T_ROT1_END:
            progress = (t - T_FADE_END) / (T_ROT1_END - T_FADE_END)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B1, angle)
            vectors = apply_motor_to_euclidean(M, initial_vectors)
        elif t < T_ROT2_END:
            # After first rotation, apply second
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            progress = (t - T_ROT1_END) / (T_ROT2_END - T_ROT1_END)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B2, angle)
            vectors = apply_motor_to_euclidean(M, v1)
        elif t < T_ROT2_END + 0.017:
            # View switch frame
            if current_projection != (1, 2, 3):
                current_projection = (1, 2, 3)
                anim.set_projection(Q, current_projection)
                anim.set_basis_labels((r"$\mathbf{e}_2$", r"$\mathbf{e}_3$", r"$\mathbf{e}_4$"))
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            vectors = apply_motor_to_euclidean(M2_full, v1)
        elif t < T_ROT3_END:
            # After two rotations, apply third
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            v2 = apply_motor_to_euclidean(M2_full, v1)
            progress = (t - T_ROT2_END) / (T_ROT3_END - T_ROT2_END)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B1, angle)
            vectors = apply_motor_to_euclidean(M, v2)
        else:
            # After three rotations, apply fourth
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            v2 = apply_motor_to_euclidean(M2_full, v1)
            M3_full = Motor.rotor(B1, TOTAL_ROTATION)
            v3 = apply_motor_to_euclidean(M3_full, v2)
            progress = min((t - T_ROT3_END) / (T_ROT4_END - T_ROT3_END), 1.0)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B2, angle)
            vectors = apply_motor_to_euclidean(M, v3)

        anim.set_vectors(Q, vectors, origin)
        anim.capture(t)

    print("Playing animation... (close window when done)")
    anim.play(loop=False)


def save_demo():
    """Save the animation as a GIF."""
    fps = 30

    # Create 4D basis and quadvector
    e1, e2, e3, e4 = basis_vectors(dim=4)
    Q = e1 ^ e2 ^ e3 ^ e4

    initial_vectors = [e1.data, e2.data, e3.data, e4.data]
    origin = zeros(4)

    B1 = create_rotation_bivector([0, 1], 2, dim=4)
    B2 = create_rotation_bivector([0, 1, 2], 3, dim=4)

    # Longer timeline for GIF
    t_fade_end = 1.0
    t_rot1_end = 4.0
    t_rot2_end = 7.0
    t_rot3_end = 11.0
    t_rot4_end = 15.0

    anim = Animation(fps=fps, theme="obsidian", size=(400, 300), show_basis=True)
    anim.track(Q, color=RED)
    anim.set_projection(Q, (0, 1, 2))
    anim.set_vectors(Q, initial_vectors, origin)
    anim.set_basis_labels((r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_3$"))
    anim.fade_in(Q, t=0.0, duration=1.0)
    anim.camera(position=(4.62, -2.64, 3.85), focal_point=(0, 0, 0))

    anim.start()
    n_frames = int(t_rot4_end * fps)
    current_projection = (0, 1, 2)

    M1_full = Motor.rotor(B1, TOTAL_ROTATION)
    M2_full = Motor.rotor(B2, TOTAL_ROTATION)

    for frame in range(n_frames + 1):
        t = frame / fps

        if t < t_fade_end:
            vectors = initial_vectors
        elif t < t_rot1_end:
            progress = (t - t_fade_end) / (t_rot1_end - t_fade_end)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B1, angle)
            vectors = apply_motor_to_euclidean(M, initial_vectors)
        elif t < t_rot2_end:
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            progress = (t - t_rot1_end) / (t_rot2_end - t_rot1_end)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B2, angle)
            vectors = apply_motor_to_euclidean(M, v1)
        elif t < t_rot2_end + 0.034:
            if current_projection != (1, 2, 3):
                current_projection = (1, 2, 3)
                anim.set_projection(Q, current_projection)
                anim.set_basis_labels((r"$\mathbf{e}_2$", r"$\mathbf{e}_3$", r"$\mathbf{e}_4$"))
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            vectors = apply_motor_to_euclidean(M2_full, v1)
        elif t < t_rot3_end:
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            v2 = apply_motor_to_euclidean(M2_full, v1)
            progress = (t - t_rot2_end) / (t_rot3_end - t_rot2_end)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B1, angle)
            vectors = apply_motor_to_euclidean(M, v2)
        else:
            v1 = apply_motor_to_euclidean(M1_full, initial_vectors)
            v2 = apply_motor_to_euclidean(M2_full, v1)
            M3_full = Motor.rotor(B1, TOTAL_ROTATION)
            v3 = apply_motor_to_euclidean(M3_full, v2)
            progress = min((t - t_rot3_end) / (t_rot4_end - t_rot3_end), 1.0)
            angle = TOTAL_ROTATION * ease_in_out_cubic(progress)
            M = Motor.rotor(B2, angle)
            vectors = apply_motor_to_euclidean(M, v3)

        anim.set_vectors(Q, vectors, origin)
        anim.capture(t)

    anim.save("figures/rotating_4blade.gif", loop=True)
    print("Saved figures/rotating_4blade.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4D blade rotation animation")
    parser.add_argument("--save", action="store_true", help="Save as GIF instead of playing")
    args = parser.parse_args()

    if args.save:
        save_demo()
    else:
        main()
