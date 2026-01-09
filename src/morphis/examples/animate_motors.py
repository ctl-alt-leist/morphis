"""
Motor Animation Demo

Demonstrates Motor rotations using proper Geometric Algebra:
- Building up basis vectors and bivectors visually
- Rotations defined by bivectors (oriented planes), not axis vectors
- Screw motions combining rotation and translation
- Timeline-based animation with smoothing

In GA, bivectors represent rotation planes directly. The bivector e1^e2
represents the xy-plane, and rotating "in" that plane is equivalent to
rotating "about" the z-axis in traditional notation.

Animation Sequence:
    Phase 1: Build up basis vectors and bivector (all red)
      [0-1s]   DRAW e1
      [1-2s]   HIDE e1, DRAW e2
      [2-3s]   HIDE e2, DRAW e12 = e1 ^ e2
      [3-4s]   DRAW e3

    Phase 2: Form the trivector
      [4-5s]   HIDE e3, REPLACE e12 -> q = e1 ^ e2 ^ e3

    Phase 3: Transformations
      [5-8s]   ROTATE q in diagonal plane by 4pi (about (1,1,1) axis)
      [8-12s]  SCREW q: rotate 2pi + translate along (1,1,1)
      [12-15s] ROTATE q into xy-plane: rotate in (e1+e2)^e3 plane by -pi/4

Run: uv run python -m morphis.examples.animate_motors
"""

from numpy import array, pi, sqrt

from morphis.ga.constructors import basis_vectors
from morphis.visualization.animated import AnimatedCanvas
from morphis.visualization.sequence import Timeline


# Red color for all trivector-related objects
RED = (0.85, 0.2, 0.2)


def main():
    """Motor animation with proper GA philosophy."""
    # Build basis vectors
    e1, e2, e3 = basis_vectors(dim=3)

    # Build bivector and trivector via wedge products
    e12 = e1 ^ e2  # xy-plane bivector
    q = e1 ^ e2 ^ e3  # trivector (unit cube)

    # Build other basis bivectors for rotation plane construction
    e23 = e2 ^ e3  # yz-plane
    e31 = e3 ^ e1  # zx-plane

    # Diagonal rotation plane: perpendicular to (1,1,1)
    # In GA, this bivector represents the plane directly
    # Sum of basis bivectors, normalized
    s = 1.0 / sqrt(3.0)
    diagonal = (e12 * s) + (e23 * s) + (e31 * s)

    # Final rotation plane: (e1 + e2) ^ e3
    # This tilts the trivector into the xy-plane
    e1_plus_e2 = e1 + e2
    final_plane = e1_plus_e2 ^ e3

    # Translation direction along (1,1,1)
    translation_dir = array([1.0, 1.0, 1.0]) / sqrt(3.0)

    # Build timeline
    timeline = Timeline()

    # =========================================================================
    # Phase 1: Build up basis vectors and bivector (all red)
    # =========================================================================

    # [0-1s] Draw e1
    timeline.draw(e1, duration=1.0, color=RED)

    # [1-2s] Hide e1 quickly, then draw e2
    timeline.hide(e1, duration=0.2)
    timeline.draw(e2, duration=0.8, color=RED)

    # [2-3s] Hide e2 quickly, then draw e12
    timeline.hide(e2, duration=0.2)
    timeline.draw(e12, duration=0.8, color=RED)

    # [3-4s] Draw e3
    timeline.draw(e3, duration=1.0, color=RED)

    # =========================================================================
    # Phase 2: Form the trivector
    # =========================================================================

    # [4-5s] Hide e3 quickly, then replace e12 with q
    timeline.hide(e3, duration=0.2)
    timeline.replace(e12, q, duration=0.8, new_color=RED)

    # =========================================================================
    # Phase 3: Transformations
    # =========================================================================

    # [5-8s] Rotate 4pi in diagonal plane (about (1,1,1) axis)
    timeline.rotate(q, plane=diagonal, angle=4 * pi, duration=3.0, smooth="in_out_cubic")

    # [8-12s] Screw motion: rotate 2pi + translate along (1,1,1)
    timeline.screw(
        q,
        plane=diagonal,
        angle=2 * pi,
        translation=translation_dir * 2.0,  # Translate 2 units along (1,1,1)
        duration=4.0,
        smooth="in_out_cubic",
    )

    # [12-15s] Rotate into xy-plane: rotate in (e1+e2)^e3 plane by -pi/4
    # Negative angle rotates in the opposite direction, bringing it toward the xy-plane
    timeline.rotate(q, plane=final_plane, angle=-pi / 4, duration=3.0, smooth="in_out_cubic")

    # Create canvas and play
    canvas = AnimatedCanvas(theme="obsidian")
    canvas.camera(position=(4, -3, 4), focal_point=(0, 0, 0))

    print(f"Animation: {timeline.total_duration:.0f}s")
    print()
    print("Phase 1: Build basis (red)")
    print("  [0-1s]   Draw e1")
    print("  [1-2s]   Hide e1, draw e2")
    print("  [2-3s]   Hide e2, draw e12")
    print("  [3-4s]   Draw e3")
    print()
    print("Phase 2: Form trivector")
    print("  [4-5s]   Hide e3, replace e12 -> q")
    print()
    print("Phase 3: Transformations")
    print("  [5-8s]   Rotate 4pi about (1,1,1)")
    print("  [8-12s]  Screw: rotate 2pi + translate")
    print("  [12-15s] Rotate -pi/4 in (e1+e2)^e3 plane (orbit)")
    print()
    print("Close window to exit.")

    canvas.play(timeline, follow=q)


if __name__ == "__main__":
    main()
