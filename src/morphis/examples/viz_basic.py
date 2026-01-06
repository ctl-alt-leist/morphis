"""
Visualization - Basic Usage Example

Demonstrates the core visualization API: arrows, curves, points, and planes.
Shows color cycling, explicit colors, and camera control.

Run with: uv run python -m morphis.examples.viz_basic
"""

from numpy import array, cos, linspace, pi, sin, stack

from morphis.visualization import Canvas


def main():
    """Basic visualization example."""
    print()
    print("=" * 50)
    print("  VISUALIZATION BASIC EXAMPLE")
    print("=" * 50)
    print()

    # Create canvas with obsidian theme
    # Basis vectors are drawn automatically (toggle with show_basis=False)
    canvas = Canvas("obsidian", title="Basic Example")

    # -------------------------------------------------------------------------
    # Arrows - color cycles automatically through the palette
    # -------------------------------------------------------------------------

    # Each arrow gets the next color in the palette
    canvas.arrow([0, 0, 0], [1.0, 0.2, 0.1])
    canvas.arrow([0, 0, 0], [0.2, 0.9, 0.3])
    canvas.arrow([0, 0, 0], [0.3, 0.1, 0.8])

    # Multiple arrows sharing one color (next palette color)
    starts = array([
        [1.0, 0.0, 0.0],
        [1.0, 0.3, 0.0],
        [1.0, 0.6, 0.0],
    ])
    directions = array([
        [0.4, 0.0, 0.0],
        [0.4, 0.1, 0.0],
        [0.4, 0.2, 0.1],
    ])
    canvas.arrows(starts, directions)

    # Explicit color using theme's accent
    canvas.arrow(
        [-0.3, -0.3, 0],
        [0.5, 0.5, 0.8],
        color=canvas.theme.accent,
        shaft_radius=0.02,
    )

    # -------------------------------------------------------------------------
    # Curves - smooth tubes through points
    # -------------------------------------------------------------------------

    # Helix curve (gets next palette color)
    t = linspace(0, 3 * pi, 80)
    helix = stack(
        [
            0.2 * cos(t) - 0.5,
            0.2 * sin(t) - 0.5,
            0.08 * t,
        ],
        axis=1,
    )
    canvas.curve(helix, radius=0.012)

    # Circle with explicit color
    theta = linspace(0, 2 * pi, 60)
    circle = stack(
        [
            0.4 * cos(theta) + 0.8,
            0.4 * sin(theta) + 0.8,
            0.5 * (1 + 0 * theta),
        ],
        axis=1,
    )
    canvas.curve(circle, color=canvas.theme.e2, radius=0.01)

    # -------------------------------------------------------------------------
    # Points - spheres at locations
    # -------------------------------------------------------------------------

    # Single point
    canvas.point([0.5, 0.5, 0.5], radius=0.04)

    # Multiple points (share one palette color)
    pts = array([
        [0.0, 1.0, 0.2],
        [0.2, 1.0, 0.4],
        [0.4, 1.0, 0.2],
    ])
    canvas.points(pts, radius=0.03)

    # Points cycling through palette colors
    canvas.points(
        [[0.8, 0.0, 0.3], [0.9, 0.0, 0.5], [1.0, 0.0, 0.7]],
        colors=canvas.theme.palette,
        radius=0.035,
    )

    # -------------------------------------------------------------------------
    # Planes - semi-transparent surfaces
    # -------------------------------------------------------------------------

    # Reference plane at z=0
    canvas.plane(
        center=[0.3, 0.3, 0.0],
        normal=[0, 0, 1],
        size=1.2,
        color=canvas.theme.muted,
        opacity=0.12,
    )

    # -------------------------------------------------------------------------
    # Camera and display
    # -------------------------------------------------------------------------

    canvas.camera(
        position=(2.8, -2.0, 2.2),
        focal_point=(0.3, 0.3, 0.4),
    )

    print("Showing visualization...")
    print("  - Rotate: left-click drag")
    print("  - Pan: middle-click drag or shift+left-click")
    print("  - Zoom: scroll wheel or right-click drag")
    print()

    canvas.show()

    print("Window closed.")


if __name__ == "__main__":
    main()
