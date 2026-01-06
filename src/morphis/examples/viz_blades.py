"""
Blade Visualization Examples

Demonstrates visualization of geometric algebra blades:
- Scalars, vectors, bivectors, trivectors
- Different rendering modes for bivectors
- PGA points, lines, planes
- Meet and join operations
- Dual visualization

Run with: uv run python -m morphis.examples.viz_blades
"""

from numpy import array

from morphis.ga.model import bivector_blade, trivector_blade, vector_blade
from morphis.ga.operations import wedge
from morphis.geometry.projective import line, plane, point
from morphis.visualization import (
    BladeStyle,
    Canvas,
    render_bivector,
    render_meet_join,
    render_with_dual,
    visualize_blade,
    visualize_blades,
    visualize_pga_blade,
    visualize_pga_scene,
)


# =============================================================================
# Demo Functions
# =============================================================================


def demo_vectors():
    """Demonstrate vector visualization."""
    print("Demo: Vectors")

    canvas = Canvas("obsidian", title="Vectors")

    # Create some vectors
    v1 = vector_blade(array([1.0, 0.0, 0.0]))
    v2 = vector_blade(array([0.0, 1.0, 0.0]))
    v3 = vector_blade(array([0.0, 0.0, 1.0]))
    v4 = vector_blade(array([0.7, 0.7, 0.5]))

    # Visualize each (colors will cycle)
    visualize_blade(v1, canvas)
    visualize_blade(v2, canvas)
    visualize_blade(v3, canvas)
    visualize_blade(v4, canvas, style=BladeStyle(color=canvas.theme.accent))

    canvas.camera(position=(3, -2, 2), focal_point=(0.3, 0.3, 0.3))
    canvas.show()


def demo_bivector_modes():
    """Demonstrate different bivector rendering modes."""
    print("Demo: Bivector Modes")

    # Create a bivector (e1 ∧ e2 plane)
    B_data = array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    B = bivector_blade(B_data)

    # Show each mode
    for mode in ["circle", "parallelogram", "plane", "circular_arrow"]:
        print(f"  Mode: {mode}")
        canvas = Canvas("obsidian", title=f"Bivector: {mode}")

        style = BladeStyle(color=canvas.theme.palette[1], scale=1.5)
        render_bivector(B, canvas, mode=mode, style=style)

        canvas.camera(position=(2, -2, 1.5), focal_point=(0, 0, 0))
        canvas.show()


def demo_trivector():
    """Demonstrate trivector visualization."""
    print("Demo: Trivector")

    canvas = Canvas("obsidian", title="Trivector (Parallelepiped)")

    # Create trivector (volume element)
    T_data = array([[[0.0] * 3] * 3] * 3)
    # Set T^{012} = 1 (and antisymmetric permutations)
    T_data[0, 1, 2] = 1.0
    T_data[0, 2, 1] = -1.0
    T_data[1, 0, 2] = -1.0
    T_data[1, 2, 0] = 1.0
    T_data[2, 0, 1] = 1.0
    T_data[2, 1, 0] = -1.0

    T = trivector_blade(T_data)

    style = BladeStyle(color=canvas.theme.palette[2], scale=0.8)
    visualize_blade(T, canvas, style=style)

    canvas.camera(position=(2.5, -2, 2), focal_point=(0.3, 0.3, 0.3))
    canvas.show()


def demo_wedge_product():
    """Demonstrate wedge product visualization."""
    print("Demo: Wedge Product (Join)")

    canvas = Canvas("obsidian", title="Wedge Product: u ∧ v")

    # Two vectors
    u = vector_blade(array([1.0, 0.2, 0.0]))
    v = vector_blade(array([0.3, 0.9, 0.2]))

    # Visualize the join operation
    render_meet_join(u, v, canvas, show="join")

    canvas.camera(position=(2, -1.5, 1.5), focal_point=(0.3, 0.3, 0.1))
    canvas.show()


def demo_meet():
    """Demonstrate meet (intersection) visualization."""
    print("Demo: Meet (Intersection)")

    canvas = Canvas("obsidian", title="Meet of Two Bivectors")

    # Two bivectors (two planes)
    u = vector_blade(array([1.0, 0.0, 0.0]))
    v = vector_blade(array([0.0, 1.0, 0.0]))
    w = vector_blade(array([0.0, 0.0, 1.0]))

    # xy-plane and xz-plane
    B1 = wedge(u, v)
    B2 = wedge(u, w)

    render_meet_join(B1, B2, canvas, show="meet")

    canvas.camera(position=(2.5, -2, 2), focal_point=(0, 0, 0))
    canvas.show()


def demo_dual():
    """Demonstrate dual visualization."""
    print("Demo: Blade with Dual")

    canvas = Canvas("obsidian", title="Vector and its Dual (Bivector)")

    # A vector
    v = vector_blade(array([1.0, 0.5, 0.3]))

    # Show vector and its dual (which is a bivector in 3D)
    render_with_dual(v, canvas, dual_type="right")

    canvas.camera(position=(2, -2, 1.5), focal_point=(0.3, 0.2, 0.1))
    canvas.show()


def demo_pga_points():
    """Demonstrate PGA point visualization."""
    print("Demo: PGA Points")

    canvas = Canvas("paper", title="PGA Points")

    # Create points at different locations
    p1 = point(array([0.0, 0.0, 0.0]))  # Origin
    p2 = point(array([1.0, 0.0, 0.0]))
    p3 = point(array([0.5, 0.8, 0.0]))
    p4 = point(array([0.5, 0.4, 0.7]))

    # Visualize
    visualize_pga_scene(p1, p2, p3, p4, canvas=canvas)

    canvas.camera(position=(2.5, -2, 2), focal_point=(0.5, 0.4, 0.3))
    canvas.show()


def demo_pga_line():
    """Demonstrate PGA line visualization."""
    print("Demo: PGA Line")

    canvas = Canvas("paper", title="PGA Line Through Two Points")

    # Two points
    p1 = point(array([0.0, 0.0, 0.0]))
    p2 = point(array([1.0, 0.5, 0.3]))

    # Line through them
    L = line(p1, p2)

    # Show points and line
    visualize_pga_blade(p1, canvas)
    visualize_pga_blade(p2, canvas)
    visualize_pga_blade(L, canvas)

    canvas.camera(position=(3, -2, 2), focal_point=(0.5, 0.25, 0.15))
    canvas.show()


def demo_pga_plane():
    """Demonstrate PGA plane visualization."""
    print("Demo: PGA Plane")

    canvas = Canvas("paper", title="PGA Plane Through Three Points")

    # Three points
    p1 = point(array([0.0, 0.0, 0.0]))
    p2 = point(array([1.0, 0.0, 0.0]))
    p3 = point(array([0.0, 1.0, 0.0]))

    # Plane through them
    H = plane(p1, p2, p3)

    # Show
    visualize_pga_blade(p1, canvas)
    visualize_pga_blade(p2, canvas)
    visualize_pga_blade(p3, canvas)
    visualize_pga_blade(H, canvas)

    canvas.camera(position=(2.5, -2, 2.5), focal_point=(0.3, 0.3, 0))
    canvas.show()


def demo_multiple_blades():
    """Demonstrate visualizing multiple blades together."""
    print("Demo: Multiple Blades")

    canvas = Canvas("midnight", title="Multiple Blades")

    # Create various blades
    blades = [
        vector_blade(array([1.0, 0.0, 0.0])),
        vector_blade(array([0.0, 0.8, 0.0])),
        vector_blade(array([0.0, 0.0, 0.6])),
        vector_blade(array([0.5, 0.5, 0.5])),
    ]

    # Visualize all (colors will cycle through palette)
    visualize_blades(blades, canvas)

    canvas.camera(position=(2.5, -2, 2), focal_point=(0.3, 0.3, 0.3))
    canvas.show()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run blade visualization demos."""
    print()
    print("=" * 60)
    print("  BLADE VISUALIZATION EXAMPLES")
    print("  Close each window to see the next demo")
    print("=" * 60)
    print()

    demos = [
        ("Vectors", demo_vectors),
        ("Bivector Modes", demo_bivector_modes),
        ("Trivector", demo_trivector),
        ("Wedge Product", demo_wedge_product),
        ("Meet Operation", demo_meet),
        ("Dual", demo_dual),
        ("PGA Points", demo_pga_points),
        ("PGA Line", demo_pga_line),
        ("PGA Plane", demo_pga_plane),
        ("Multiple Blades", demo_multiple_blades),
    ]

    for name, demo_func in demos:
        print(f"\n--- {name} ---")
        demo_func()

    print()
    print("All demos complete!")
    print()


if __name__ == "__main__":
    main()
