"""
Grover's Algorithm — Geometric Visualization

Visualizes the geometry of Grover's search algorithm using geometric algebra:
- The |good⟩/|bad⟩ plane as a 2D rotation space
- Oracle reflection: phase flip as reflection across |bad⟩ axis
- Diffusion operator: reflection across uniform superposition |s⟩
- Two reflections compose into rotation by 2θ toward |good⟩
- State vector sweeping toward the solution with each iteration

The key insight: Grover's algorithm is a sequence of rotor operations
in the plane spanned by the target state and its complement. The oracle
and diffusion operator are each reflections; their composition is a rotation
by 2θ, where θ = arcsin(√(M/N)).

Parametric: set N_STATES and M_MARKED to explore how the geometry changes.
Default: 12 marked states out of 64 (the shift scheduling oracle).

Run: uv run python -m morphis.examples.grovers_geometry
"""

from numpy import arcsin, cos, diff, pi, sin, sqrt

from morphis.elements import Frame, basis_vectors, euclidean_metric
from morphis.operations import unit
from morphis.transforms import rotor
from morphis.utils.easing import ease_in_out_cubic
from morphis.visuals import SMALL_SQUARE, Scene


# =============================================================================
# Configuration
# =============================================================================

# Problem parameters (default: shift scheduling oracle)
N_STATES = 64  # Total states in search space (2^n)
M_MARKED = 12  # Number of marked (solution) states

# Animation timing
FRAME_RATE = 60
DURATION_FADE_IN = 1.0
DURATION_ORACLE = 1.5
DURATION_DIFFUSION = 1.5
PAUSE_BETWEEN = 0.3
PAUSE_AFTER_ITERATION = 0.8
HOLD_FINAL = 2.0

# Colors
COLOR_STATE = (0.9, 0.25, 0.2)  # State vector — warm red


# =============================================================================
# Grover Geometry
# =============================================================================


def grover_angle(n_states: int, m_marked: int) -> float:
    """
    Grover angle θ = arcsin(√(M/N)).

    Initial angle between uniform superposition |s⟩ and |bad⟩.
    Each iteration rotates the state by 2θ toward |good⟩.
    """
    return arcsin(sqrt(m_marked / n_states))


def optimal_iterations(n_states: int, m_marked: int) -> int:
    """
    Optimal number of Grover iterations.

    Rather than rounding π/4 × √(N/M), we check both floor and ceiling
    and pick whichever gives higher probability. The continuous formula
    is an approximation; the true optimum is the k that maximizes
    sin²((2k+1)θ).
    """
    from math import ceil, floor

    theta = grover_angle(n_states, m_marked)
    k_approx = pi / 4 * sqrt(n_states / m_marked)
    k_floor = max(1, floor(k_approx))
    k_ceil = max(1, ceil(k_approx))

    p_floor = probability_after(theta, k_floor)
    p_ceil = probability_after(theta, k_ceil)

    return k_floor if p_floor >= p_ceil else k_ceil


def probability_after(theta: float, k: int) -> float:
    """Probability of measuring a marked state after k iterations."""
    return sin((2 * k + 1) * theta) ** 2


# =============================================================================
# Animation Helpers
# =============================================================================


def compute_delta_angles(duration: float, total_angle: float) -> list[float]:
    """Pre-compute incremental angles for eased rotation."""
    n = int(duration * FRAME_RATE)
    angles = [total_angle * ease_in_out_cubic((i + 1) / n) for i in range(n)]
    return list(diff([0.0] + angles))


def capture_frames(scene, t: float, duration: float) -> float:
    """Capture static frames for a pause duration. Returns updated time."""
    dt = 1.0 / FRAME_RATE
    for _ in range(int(duration * FRAME_RATE)):
        scene.capture(t)
        t += dt
    return t


# =============================================================================
# Scene Construction
# =============================================================================


def create_scene(n_states: int = N_STATES, m_marked: int = M_MARKED):
    """
    Create the Grover's algorithm geometry animation.

    The algorithm is visualized as rotations in the 2D plane spanned by:
    - e1 = |bad⟩  (complement of solution space)
    - e2 = |good⟩ (uniform superposition over solutions)

    The uniform superposition |s⟩ starts at angle θ from |bad⟩. Each Grover
    iteration consists of two reflections:
    1. Oracle:    reflect across |bad⟩ axis (negate |good⟩ component)
    2. Diffusion: reflect across |s⟩ direction (reflection about mean)

    Their composition is a rotation by 2θ toward |good⟩.

    Args:
        n_states: Total number of states in search space.
        m_marked: Number of marked (solution) states.

    Returns:
        Configured Scene ready for show() or export().
    """
    theta = grover_angle(n_states, m_marked)
    k_opt = optimal_iterations(n_states, m_marked)
    scale = 1.5

    # Report
    print("Grover's Algorithm — Geometric Visualization")
    print("=" * 50)
    print(f"Search space:       N = {n_states}")
    print(f"Marked states:      M = {m_marked}")
    print(f"Grover angle:       θ = {theta:.4f} rad ({theta * 180 / pi:.1f}°)")
    print(f"Optimal iterations: k = {k_opt}")
    print()
    for i in range(k_opt + 1):
        p = probability_after(theta, i)
        bar = "█" * int(p * 40)
        print(f"  k={i}: P = {p * 100:5.1f}%  {bar}")
    print()

    # =========================================================================
    # Geometric algebra setup
    #
    # We use 3D Euclidean space but keep all vectors in the e1-e2 plane.
    # This ensures PyVista renders proper 3D arrow meshes.
    # e1 = |bad⟩ direction, e2 = |good⟩ direction, e3 unused.
    # =========================================================================

    g = euclidean_metric(3)
    e1, e2, _ = basis_vectors(g)

    # Rotation bivector (rotation in the e1-e2 plane)
    B = unit(e1 ^ e2)

    # Reference frames (static)
    bad_axis = Frame(e1 * scale)  # |bad⟩ direction
    good_axis = Frame(e2 * scale)  # |good⟩ direction

    # Initial superposition |s⟩ at angle θ from |bad⟩
    s_dir = e1 * cos(theta) + e2 * sin(theta)
    s_ref = Frame(s_dir * (scale * 0.85))  # Slightly shorter, reference line

    # State vector — this is what we animate
    state = Frame(s_dir * scale)

    # =========================================================================
    # Scene setup
    # =========================================================================

    scene = Scene(
        frame_rate=FRAME_RATE,
        theme="obsidian",
        size=SMALL_SQUARE,
    )

    scene.add(bad_axis, color=(0.4, 0.45, 0.5))
    scene.add(good_axis, color=(0.2, 0.75, 0.4))
    scene.add(s_ref, color=(0.45, 0.45, 0.45))
    scene.add(state, color=COLOR_STATE)

    # Fade in all elements
    for element in [bad_axis, good_axis, s_ref, state]:
        scene.fade_in(element, t=0.0, duration=DURATION_FADE_IN)

    # Camera: slightly elevated view looking down at the xy-plane
    scene.camera(
        position=(0, -2, 6),
        focal_point=(0.3, 0.3, 0),
    )

    t = 0.0
    dt = 1.0 / FRAME_RATE

    # Capture fade-in
    for _ in range(int(DURATION_FADE_IN * FRAME_RATE) + 1):
        scene.capture(t)
        t += dt

    # Brief hold before iterations begin
    t = capture_frames(scene, t, 0.5)

    # =========================================================================
    # Grover iterations
    # =========================================================================

    current_angle = theta

    for iteration in range(k_opt):
        # -----------------------------------------------------------------
        # Oracle: reflect across |bad⟩ (e1 axis)
        #
        # State at angle α → reflected to -α
        # This is a rotation by -2α in the e1e2 plane
        # -----------------------------------------------------------------

        oracle_rotation = -2 * current_angle
        d_angles = compute_delta_angles(DURATION_ORACLE, oracle_rotation)

        for d_angle in d_angles:
            M = rotor(B, d_angle)
            state.data[...] = state.transform(M).data
            scene.capture(t)
            t += dt

        current_angle = -current_angle

        t = capture_frames(scene, t, PAUSE_BETWEEN)

        # -----------------------------------------------------------------
        # Diffusion: reflect across |s⟩ (initial superposition at angle θ)
        #
        # State at angle α → reflected to 2θ - α
        # This is a rotation by (2θ - α) - α = 2(θ - α)
        # -----------------------------------------------------------------

        target_angle = 2 * theta - current_angle
        diffusion_rotation = target_angle - current_angle

        d_angles = compute_delta_angles(DURATION_DIFFUSION, diffusion_rotation)

        for d_angle in d_angles:
            M = rotor(B, d_angle)
            state.data[...] = state.transform(M).data
            scene.capture(t)
            t += dt

        current_angle = target_angle

        # Report iteration result
        p = probability_after(theta, iteration + 1)
        print(
            f"  Iteration {iteration + 1}: angle = {current_angle * 180 / pi:.1f}° from |bad⟩, P(good) = {p * 100:.1f}%"
        )

        t = capture_frames(scene, t, PAUSE_AFTER_ITERATION)

    # Hold final state
    t = capture_frames(scene, t, HOLD_FINAL)

    return scene


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the Grover geometry visualization."""
    print()

    # Default: the shift scheduling oracle (Problem 2i)
    # 12 valid shift assignments out of 64 possible states
    scene = create_scene(n_states=64, m_marked=12)

    print()
    print("Close window or press 'q' to exit.")
    print()

    scene.show()


if __name__ == "__main__":
    main()
