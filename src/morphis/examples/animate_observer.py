"""
Animation with External Transforms

Demonstrates the clean separation of concerns:
1. External functions/sequences modify blade.visual_transform
2. Canvas just reads and renders on update()

Sequence: rotate 4π (8s) → translate to (1,1,1) (4s) → rotate 2π (4s)

Run with: uv run python -m morphis.examples.animate_observer
"""

import time as time_module

from numpy import array, pi, sqrt, zeros

from morphis.core.rotations import rotation_matrix
from morphis.ga.model import trivector_blade
from morphis.visualization.animated import AnimatedCanvas
from morphis.visualization.transforms import (
    AnimationSequence,
    ease_in_out_cubic,
    ease_linear,
)


def create_e123():
    """Create the unit pseudoscalar e123 = e1 ^ e2 ^ e3."""
    data = zeros((3, 3, 3))
    data[0, 1, 2] = 1.0
    data[0, 2, 1] = -1.0
    data[1, 0, 2] = -1.0
    data[1, 2, 0] = 1.0
    data[2, 0, 1] = 1.0
    data[2, 1, 0] = -1.0
    return trivector_blade(data)


def apply_sequence_to_blade(blade, sequence, time: float):
    """
    Apply animation sequence state to blade's visual transform.

    This reads the sequence's transform at the given time and applies
    it to the blade's visual_transform.
    """
    transform = sequence.evaluate(time)

    # Build rotation matrix from axis-angle
    R = rotation_matrix(transform.angle, transform.axis)

    # Apply to blade's visual transform
    blade.visual_transform.rotation = R
    blade.visual_transform.translation = array(transform.translation)


def main():
    """Interactive animation with sequenced transforms."""
    # Create the trivector
    blade = create_e123()

    # Set up canvas
    canvas = AnimatedCanvas(theme="obsidian")
    canvas.track(blade)
    # Center view between start (origin) and end (1,1,1) positions
    canvas.camera(position=(6, -4, 5), focal_point=(1.0, 1.0, 1.0))

    # Build animation sequence
    rotation_axis = (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0))

    sequence = AnimationSequence()
    sequence.rotate(
        angle=4 * pi,  # 2 full rotations
        axis=rotation_axis,
        duration=8.0,
        easing=ease_linear,  # Constant speed rotation
    )
    sequence.translate(
        target=(1.0, 1.0, 1.0),
        duration=4.0,
        easing=ease_in_out_cubic,  # Smooth start/stop
    )
    sequence.rotate(
        angle=2 * pi,  # 1 full rotation
        axis=rotation_axis,
        duration=4.0,
        center=(1.0, 1.0, 1.0),  # Rotate about new position
        easing=ease_linear,
    )

    total_duration = sequence.total_duration
    fps = 60

    print(f"Animation duration: {total_duration} seconds")
    print("  Phase 1: 4π rotation (8s)")
    print("  Phase 2: translate to (1,1,1) (4s)")
    print("  Phase 3: 2π rotation about (1,1,1) (4s)")
    print()
    print("Close window to exit.")

    # Show window
    canvas.show()

    # Animation loop
    start_time = time_module.time()
    while True:
        elapsed = time_module.time() - start_time

        if elapsed > total_duration:
            # Keep showing final state
            elapsed = total_duration

        # External: apply sequence state to blade
        apply_sequence_to_blade(blade, sequence, elapsed)

        # Canvas: just read and render
        canvas.update()

        # Timing
        time_module.sleep(1.0 / fps)

        # Check if past end
        if elapsed >= total_duration:
            print("Animation complete! Close window to exit.")
            break

    # Hand control to VTK interactor so window stays open and responsive
    canvas._plotter.iren.interactor.Start()


def main_save(filename: str = "trivector_sequence.mp4"):
    """Save animation to file."""
    blade = create_e123()

    canvas = AnimatedCanvas(theme="obsidian")
    canvas.track(blade)
    canvas.camera(position=(6, -4, 5), focal_point=(1.0, 1.0, 1.0))

    rotation_axis = (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0))

    sequence = AnimationSequence()
    sequence.rotate(angle=4 * pi, axis=rotation_axis, duration=8.0, easing=ease_linear)
    sequence.translate(target=(1.0, 1.0, 1.0), duration=4.0, easing=ease_in_out_cubic)
    sequence.rotate(
        angle=2 * pi,
        axis=rotation_axis,
        duration=4.0,
        center=(1.0, 1.0, 1.0),
        easing=ease_linear,
    )

    total_duration = sequence.total_duration
    fps = 60
    n_frames = int(total_duration * fps)

    print(f"Rendering {n_frames} frames to {filename}...")

    canvas.start_recording(filename, fps=fps)
    canvas._ensure_plotter()
    canvas._plotter.off_screen = True

    for frame in range(n_frames):
        t = frame / fps

        apply_sequence_to_blade(blade, sequence, t)
        canvas.update()

        if (frame + 1) % (fps * 2) == 0:
            print(f"  {t:.1f}s / {total_duration:.1f}s")

    canvas.stop_recording()
    print(f"Saved to {filename}")


if __name__ == "__main__":
    import sys

    if "--save" in sys.argv:
        filename = "trivector_sequence.mp4"
        for arg in sys.argv:
            if arg.endswith(".mp4"):
                filename = arg
                break
        main_save(filename)
    else:
        main()
