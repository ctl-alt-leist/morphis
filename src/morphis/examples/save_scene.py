"""
Save Scene Example

Creates a static scene with basis vectors and a bivector, then saves it.

Run: uv run python -m morphis.examples.save_scene
View: morphis view figures/basis_bivector.scene
"""

from pathlib import Path

from morphis.elements import basis_vectors, euclidean_metric
from morphis.visuals import BLUE, GREEN, ORANGE, RED, Scene


def main() -> None:
    """Create and save a demo scene."""
    # Create GA objects
    g = euclidean_metric(3)
    e1, e2, e3 = basis_vectors(g)

    # Create scene
    scene = Scene(theme="obsidian")

    # Add basis vectors
    scene.add(e1, color=RED)
    scene.add(e2, color=GREEN)
    scene.add(e3, color=BLUE)

    # Add a bivector (e1 ∧ e2 plane)
    scene.add(e1 ^ e2, color=ORANGE)

    # Ensure figures directory exists
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Save both formats
    scene_path = figures_dir / "basis_bivector.scene"
    obj_path = figures_dir / "basis_bivector.obj"

    scene.save(scene_path)
    scene.save(obj_path)

    print(f"Saved: {scene_path}")
    print(f"Saved: {obj_path}")
    print()
    print(f"View with: morphis view {scene_path}")
    print(f"Or open {obj_path} in Preview/3D app")


if __name__ == "__main__":
    main()
